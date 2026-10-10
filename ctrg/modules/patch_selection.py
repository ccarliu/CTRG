import torch
import torch.nn as nn
import torch.nn.functional as F



class DefaultSelector_(nn.Module):
    """
    默认 selector: 线性投影 + 归一化点积 + 小 MLP 残差
    forward:
    query: [d] or [d_q]  (若为多维，会按第0维循环)
    cand_feats: [M_actual, d]
    cls_prior: [M_actual] or None
    返回 logits: [M_actual]
    """
    def __init__(self, d_feat=768, d_query=None, proj_dim=256, use_prior=True, tau=0.07):
        super().__init__()
        if d_query is None:
            d_query = d_feat
        self.wp = nn.Linear(d_feat, proj_dim)
        self.wt = nn.Linear(d_query, proj_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj_dim * 2 + (1 if use_prior else 0)),
            nn.Linear(proj_dim * 2 + (1 if use_prior else 0), proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
            )
        self.use_prior = use_prior
        self.tau = tau

    def forward(self, query, cand_feats, cls_prior=None):
        # cand_feats: [M, d_feat]
        # query: [d_query] or [1, d_query]
        # returns logits: [M]
        M = cand_feats.shape[0]
        p = F.normalize(self.wp(cand_feats), dim=-1)  # [M,proj_dim]
        q = F.normalize(self.wt(query.unsqueeze(0) if query.dim()==1 else query), dim=-1)  # [1,proj_dim] or [B,proj]
        q = q.squeeze(0)  # [proj_dim]
        dot = torch.matmul(p, q) / self.tau  # [M]
        # mlp residual
        q_exp = q.unsqueeze(0).expand(M, -1)  # [M,proj_dim]
        if cls_prior is None:
            prior = torch.zeros(M, 1, device=cand_feats.device)
        else:
            prior = cls_prior.unsqueeze(1)
        mlp_in = torch.cat([p, q_exp, prior], dim=-1)  # [M, 2proj + 1]
        mlp_logits = self.mlp(mlp_in).squeeze(-1)  # [M]
        logits = dot + mlp_logits
        return logits

class DefaultSelector(torch.nn.Module):
    def __init__(self, d_feat, d_query=None, proj_dim=512, use_prior=True, tau=0.07):
        super().__init__()
        if d_query is None:
            d_query = d_feat
        self.wp = torch.nn.Linear(d_feat, proj_dim)
        self.wt = torch.nn.Linear(d_query, proj_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(proj_dim * 2 + (1 if use_prior else 0)),
            torch.nn.Linear(proj_dim * 2 + (1 if use_prior else 0), proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, 1)
        )
        self.use_prior = use_prior
        self.tau = tau

    def forward(self, query, cand_feats, cls_prior=None):
        Mloc = cand_feats.shape[0]
        p = F.normalize(self.wp(cand_feats), dim=-1)
        q = F.normalize(self.wt(query.unsqueeze(0) if query.dim()==1 else query), dim=-1)
        q = q.squeeze(0)
        dot = torch.matmul(p, q) / self.tau
        q_exp = q.unsqueeze(0).expand(Mloc, -1)
        if cls_prior is None:
            prior = torch.zeros(Mloc, 1, device=cand_feats.device)
        else:
            prior = cls_prior.unsqueeze(1)
        mlp_in = torch.cat([p, q_exp, prior], dim=-1)
        mlp_logits = self.mlp(mlp_in).squeeze(-1)
        logits = dot + mlp_logits
        return logits


class DotPriorSelector(nn.Module):
    def __init__(self, d_feat, d_query=None, use_prior=True):
        super().__init__()
        d_query = d_query or d_feat
        self.use_prior = use_prior
        self.wp = nn.Linear(d_feat, d_query)   # 只做投影对齐
        self.tau = nn.Parameter(torch.tensor(0.5))

    def forward(self, query, cand_feats, cls_prior=None):
        p = F.normalize(self.wp(cand_feats), dim=-1)
        q = F.normalize(query, dim=-1)
        dot = torch.matmul(p, q) / self.tau.clamp(0.3, 1.0)
        if self.use_prior and cls_prior is not None:
            logits = dot + cls_prior * 0.5
        else:
            logits = dot
        return logits

def select_features_network_variable_k_gpu(
    features,
    attn_map,
    query,
    M=100,
    lambda_mmr=0.99,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1,
    selector_model=None,
    training=False,
    return_details=False,
    device=None
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector (可训练).
    新增参数:
    selector_model: nn.Module or None. 若为 None 会临时使用 DefaultSelector（注意：若训练时希望学习 selector，请在外部实例化并传入同一个对象）
    training: bool, 若 True 会计算 STE surrogate 等并返回额外信息（如果 return_details True）
    return_details: bool, 如果 True 返回 (selected, details) 其中 details 包含 p/topk_idx/soft_agg 等
    原有输入:
    features: [bs, P, d] (torch.Tensor on device)
    attn_map: [bs, T, P]
    query:    [bs, T, d_q]  (文本 token embedding per token)
    返回:
    selected: list(bs) -> list(T) -> Tensor [1 + k_cur, d]
    若 return_details: 还会返回 details: dict with lists for p, indices, surrogate_weights, soft_agg
    注意:
    - 若要训练 selector，请确保传入同一个 selector_model 实例（不要传 None，每个调用会重建权重）。
    - 该函数不会自动对 selector optimizer 做 step，需在外部构造 loss 并 backward。
    """
    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    # If no selector provided, create a default one (caller beware: weights are new)
    if selector_model is None:
        selector_model = DefaultSelector_(d_feat=d, d_query=query.shape[2], proj_dim=min(256, d), use_prior=True, tau=0.07)
        selector_model.to(device)

    selector_model.eval() if not training else selector_model.train()

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    # optional details containers
    details = {
        'p': [[None for _ in range(T)] for _ in range(bs)],
        'indices': [[None for _ in range(T)] for _ in range(bs)],
        'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
        'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
    } if return_details else None

    # loop over batch & tokens (GPU-friendly single-threaded)
    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()
            cand_scores = scores[idx_candidates]  # descending by construction

            if False:
                # 决定 k_cur
                if method == 'cumulative':
                    cum = torch.cumsum(cand_scores, dim=0)
                    pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                    if pos.numel() == 0:
                        n_req = M_actual
                    else:
                        n_req = int(pos[0].item()) + 1
                    k_cur = max(min_k, min(n_req, max_k, M_actual))
                elif method == 'perplexity':
                    p_dist = cand_scores / (cand_scores.sum() + eps)
                    entropy = -(p_dist * (p_dist + eps).log()).sum()
                    perp = torch.exp(entropy).item()
                    k_cur = int(round(perp))
                    k_cur = max(min_k, min(k_cur, max_k, M_actual))
                elif method == 'attn_mass':
                    mass = cand_scores.sum().item()
                    k_cur = int(round(min_k + (max_k - min_k) * mass))
                    k_cur = max(min_k, min(k_cur, max_k, M_actual))
                else:
                    raise ValueError("Unknown method")

                k_cur = max(k_cur, min_token_num)
                k_cur = min(k_cur, M_actual)
            else:
                k_cur = 10

            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留
            if M_actual <= k_cur:
                kept_idx = idx_candidates
                # For details, p=ones
                if return_details:
                    details['p'][bi][ti] = torch.ones(M_actual, device=device)
                    details['indices'][bi][ti] = kept_idx
                    details['surrogate_weights'][bi][ti] = torch.ones(M_actual, device=device)
                    details['soft_agg'][bi][ti] = torch.matmul(details['surrogate_weights'][bi][ti], feats_b[kept_idx])
                token_patches = feats_b[kept_idx]  # [k_cur, d]
                token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
                batch_list.append(token_feats)
                continue

            # ####################################
            # 用 selector 在候选池上打分（可训练 head）
            # ####################################
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cls_prior = cand_scores  # 可以直接用 cand_scores 作为 prior（注意范围是 [0,1]）
            # selector_model forward expects query vector and cand_feats
            logits = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            # logits -> soft scores
            p = torch.sigmoid(logits)  # [M_actual], 可导

            # top-k on p
            topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)  # positions in [0..M_actual)
            kept_idx = idx_candidates[topk_pos]  # original indices in [0..P)
            # hard multi-hot mask in candidate space
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            # straight-through surrogate
            if training:
                y = (hard - p).detach() + p  # [M_actual] surrogate weights (forward looks like hard, backward p)
            else:
                y = hard  # inference: use hard weights (no grad)

            # optionally: soft aggregation for loss
            soft_agg = torch.matmul(y, cand_feats)  # [d]  (可微 if training)

            # ensure at least min_token_num by filling from remaining by attn if needed
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回的 token_feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

            # fill details
            if return_details:
                details['p'][bi][ti] = p.detach() if not training else p  # 注意 training 时希望保留 grad -> 不 detach
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y.detach() if not training else y
                details['soft_agg'][bi][ti] = soft_agg.detach() if not training else soft_agg

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

class DefaultSelector_2(nn.Module):
    def __init__(self, d_feat, d_query=None, proj_dim=512, use_prior=True):
        super().__init__()
        d_query = d_query or d_feat
        self.use_prior = use_prior

        # 投影层
        self.wp = nn.Linear(d_feat, proj_dim)
        self.wt = nn.Linear(d_query, proj_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )

        # 可学习温度
        self.tau = nn.Parameter(torch.tensor(0.5))

        # dot 分支 LayerNorm
        self.norm_dot = nn.LayerNorm(1)

        self.weight = nn.Linear(3, 1, bias=False)
        # 初始平均加权，避免一开始极端
        with torch.no_grad():
            self.weight.weight[:] = torch.tensor([1.0, 1.0, 1.0])

    def forward(self, query, cand_feats, cls_prior=None):
        Mloc = cand_feats.shape[0]
        device = cand_feats.device

        # 1. 投影 + 归一化
        p = F.normalize(self.wp(cand_feats), dim=-1)
        q = F.normalize(self.wt(query.unsqueeze(0) if query.dim() == 1 else query),
                        dim=-1).squeeze(0)

        # 2. scaled dot + 温度 + LayerNorm
        tau = self.tau.clamp(0.3, 1.0)
        dot = torch.matmul(p, q) / tau
        # dot = self.norm_dot(dot.unsqueeze(-1)).squeeze(-1)
        # print("xxk")
        return dot

        # 3. MLP 分支
        q_exp = q.unsqueeze(0).expand(Mloc, -1)
        mlp_logits = self.mlp(torch.cat([p, q_exp], dim=-1)).squeeze(-1)

        # 4. prior bias
        if self.use_prior and cls_prior is not None:
            prior_bias = cls_prior * 0.5
        else:
            prior_bias = torch.zeros(Mloc, device=device)

        three = torch.stack([dot, mlp_logits, prior_bias], dim=1)  # [M, 3]
        logits = self.weight(three).squeeze(1)                     # [M]

        # print(logits, logits.shape)
        return logits

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_topk_relaxed(logits, K, tau=0.5, training=True, eps=1e-12):
    """
    logits: 1D tensor [M]
    K: int <= M
    tau: fixed temperature (float)
    training: bool -> if True use Gumbel perturb + relaxed soft K-hot (STE surrogate)
                     if False deterministic hard top-k (no noise)
    Returns:
      y: surrogate weights (sum == K); in eval this is exact K-hot.
      hard: exact K-hot mask (0/1)
      topk_idx: indices of selected positions
    """
    M = logits.numel()
    K = min(int(K), M)
    device = logits.device

    if K <= 0:
        return torch.zeros_like(logits, device=device), torch.zeros_like(logits, device=device), torch.empty(0, dtype=torch.long, device=device)

    if training:
        # 1) sample Gumbel and perturb logits
        g = sample_gumbel((M,), device=device)
        logits_pert = logits + g

        # 2) soft relaxed K-hot: softmax over perturbed logits scaled to sum==K
        y_soft = F.softmax(logits_pert / (tau + eps), dim=0) * float(K)  # sum == K

        # 3) hard top-k according to perturbed logits (sample without replacement)
        topk_vals, topk_idx = torch.topk(logits_pert, K, largest=True)
        hard = torch.zeros_like(logits, device=device)
        hard[topk_idx] = 1.0

        # 4) STE: forward uses hard, backward uses y_soft
        y = hard.detach() - y_soft.detach() + y_soft
        return y, hard, topk_idx
    else:
        # deterministic evaluation: top-k on original logits
        topk_vals, topk_idx = torch.topk(logits, K, largest=True)
        hard = torch.zeros_like(logits, device=device)
        hard[topk_idx] = 1.0
        return hard, hard, topk_idx

def select_features_network_mmr_variable_k_gpu_purefeature_2(
    features, attn_map, query, selector_model, M=100, lambda_mmr=1.0, tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False,
    return_details=False, device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True, beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector + 可微软化 MMR.
    新增 soft-MMR 参数:
    enable_soft_mmr: bool, 是否启用可微软化 MMR（默认 True）
    beta_mmr: float, redundancy 惩罚系数（越大越鼓励多样性）
    mmr_exclude_self: bool, 是否在 S 中置零对角（避免 self contribute）
    mmr_normalize_redundancy: bool, 是否把 redundancy 标准化（/sum(p0) 或 /max）以更稳定调参

    返回与之前一致；若 return_details=True，会在 details 中额外返回 'logits_raw' 和 'logits_mmr' 供分析。
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []
        # print("xxxxxxxxxxxxxxx")
        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            idx_candidates = torch.arange(P, device=device)
            
            cand_scores = scores[idx_candidates]
            order = torch.argsort(cand_scores, descending=True)
            idx_candidates = idx_candidates[order][:M]


            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选，返回仅 query
                batch_list.append([])
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates]  # descending by construction
            # print(scores.max(), scores.min())
            k_cur = 10

            # ########################
            # 用 selector 在候选池上打分（可训练 head）
            # ########################
            # print(idx_candidates)
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]  # [M_actual, d] (已归一化)
            S = torch.matmul(cand_feats_norm, cand_feats_norm.t()) 

            cls_prior = cand_scores  # prior
            # print(cls_prior)
            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            # print(logits_raw.shape)

            # print(cand_scores.sum())

            logits_mmr = logits_raw

            if enable_soft_mmr and M_actual > 1 and False: 
                # 计算相似矩阵 S (cosine), 排除对角可选
                S = torch.matmul(cand_feats_norm, cand_feats_norm.t())  # [M_actual, M_actual]

                
                if mmr_exclude_self:
                    # 把对角置0，避免 self contribute
                    try:
                        S = S.clone()
                        S.fill_diagonal_(0.0)
                    except Exception:
                        # 兼容旧 pytorch: 手动减去对角
                        diag = torch.diag(torch.diag(S))
                        S = S - diag

                # base selection probability
                p0 = torch.sigmoid(logits_raw)  # [M_actual]
                # redundancy: S @ p0
                redundancy = torch.matmul(S, p0)  # [M_actual]

                if mmr_normalize_redundancy:
                    # 标准化 redundancy，避免 scale 问题：常见选择是 / (p0.sum()+eps) 或 / redundancy.max()
                    denom = p0.sum().clamp(min=eps)
                    redundancy = redundancy / denom
                    # 也可用 max normalization： redundancy = redundancy / (redundancy.max()+eps)

                # 调整 logits
                logits_mmr = logits_raw - beta_mmr * redundancy
            # print(logits_mmr)
            # logits -> soft scores

            # 替换前（示意）：
            # p = torch.softmax(logits_mmr, 0)
            # topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)
            # hard = torch.zeros_like(p); hard[topk_pos] = 1.0
            # if training: y = (hard - p).detach() + p
            # else: y = hard

            # 替换后：用固定 tau（例如 tau_selector=0.5）
            tau_selector = 0.5  # 训练时/调试时可以改为 0.3/0.2 或 0.8 等
            y, hard, topk_pos = gumbel_topk_relaxed(logits_mmr, K=k_cur, tau=tau_selector, training=training)
            # 使用 y 计算 soft_agg、surrogate_weights 等
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            kept_idx = idx_candidates[topk_pos]  # original global indices

            # soft aggregation
            # soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = token_patches
            batch_list.append(token_feats)

            # fill details
            if return_details:
                # details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_network_mmr_variable_k_gpu_purefeature_r(
    features, attn_map, query, selector_model, M=100, lambda_mmr=1.0, tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False, k_cur = 10,
    return_details=False, device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True, beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector + 可微软化 MMR.
    新增 soft-MMR 参数:
    enable_soft_mmr: bool, 是否启用可微软化 MMR（默认 True）
    beta_mmr: float, redundancy 惩罚系数（越大越鼓励多样性）
    mmr_exclude_self: bool, 是否在 S 中置零对角（避免 self contribute）
    mmr_normalize_redundancy: bool, 是否把 redundancy 标准化（/sum(p0) 或 /max）以更稳定调参

    返回与之前一致；若 return_details=True，会在 details 中额外返回 'logits_raw' 和 'logits_mmr' 供分析。
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []
        # print("xxxxxxxxxxxxxxx")
        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            idx_candidates = torch.arange(P, device=device)
            
            cand_scores = scores[idx_candidates]
            order = torch.argsort(cand_scores, descending=True)
            idx_candidates = idx_candidates[order][:M]


            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选，返回仅 query
                batch_list.append([])
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates]  # descending by construction
            # print(scores.max(), scores.min())
            

            # ########################
            # 用 selector 在候选池上打分（可训练 head）
            # ########################
            # print(idx_candidates)
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]  # [M_actual, d] (已归一化)
            S = torch.matmul(cand_feats_norm, cand_feats_norm.t()) 

            cls_prior = cand_scores  # prior
            # print(cls_prior)
            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            # print(logits_raw.shape)

            # print(cand_scores.sum())

            logits_mmr = logits_raw

            if enable_soft_mmr and M_actual > 1 and False: 
                # 计算相似矩阵 S (cosine), 排除对角可选
                S = torch.matmul(cand_feats_norm, cand_feats_norm.t())  # [M_actual, M_actual]

                
                if mmr_exclude_self:
                    # 把对角置0，避免 self contribute
                    try:
                        S = S.clone()
                        S.fill_diagonal_(0.0)
                    except Exception:
                        # 兼容旧 pytorch: 手动减去对角
                        diag = torch.diag(torch.diag(S))
                        S = S - diag

                # base selection probability
                p0 = torch.sigmoid(logits_raw)  # [M_actual]
                # redundancy: S @ p0
                redundancy = torch.matmul(S, p0)  # [M_actual]

                if mmr_normalize_redundancy:
                    # 标准化 redundancy，避免 scale 问题：常见选择是 / (p0.sum()+eps) 或 / redundancy.max()
                    denom = p0.sum().clamp(min=eps)
                    redundancy = redundancy / denom
                    # 也可用 max normalization： redundancy = redundancy / (redundancy.max()+eps)

                # 调整 logits
                logits_mmr = logits_raw - beta_mmr * redundancy
            # print(logits_mmr)
            # logits -> soft scores
            p = torch.softmax(logits_mmr, 0)  # [M_actual]
            # print(p)
            # top-k on p
            topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)  # positions in candidate pool
            # print(topk_vals, topk_pos)
            kept_idx = idx_candidates[topk_pos]  # original global indices

            # hard mask & STE surrogate
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            if training:
                y = (hard - p).detach() + p
            else:
                y = hard


            # soft aggregation
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = token_patches
            batch_list.append(token_feats)

            # fill details
            if return_details:
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_network_mmr_variable_k_gpu_purefeature_r_vis(
    features, attn_map, query, selector_model, M=100, lambda_mmr=1.0, tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False, k_cur = 10,
    return_details=False, device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True, beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
    ):
    """
    修改说明：
    现在除了返回 selected (特征向量) 外，还会返回 selected_indices (在原始 features 中的索引)。
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    
    selected = []
    # [新增 1] 初始化索引列表容器
    selected_indices = [] 
    
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        
        batch_list = []
        # [新增 2] 初始化当前 batch 的索引列表
        batch_indices_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            idx_candidates = torch.arange(P, device=device)
            
            cand_scores = scores[idx_candidates]
            order = torch.argsort(cand_scores, descending=True)
            idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选
                batch_list.append([])
                # [新增 3] 处理空候选情况，保持结构一致
                batch_indices_list.append(torch.empty(0, dtype=torch.long, device=device))
                
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates] 

            # ########################
            # 用 selector 在候选池上打分
            # ########################
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]
            
            cls_prior = cand_scores
            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]

            logits_mmr = logits_raw

            # (此处省略了被禁用的 MMR 逻辑代码，保持原样即可)
            if enable_soft_mmr and M_actual > 1 and False: 
                 # ... 原有 MMR 逻辑 ...
                 pass

            # logits -> soft scores
            p = torch.softmax(logits_mmr, 0)  # [M_actual]
            
            # top-k on p
            topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)  # positions in candidate pool
            
            # [关键点] kept_idx 就是在原始 features (0~P-1) 中的索引
            kept_idx = idx_candidates[topk_pos]  # original global indices

            # hard mask & STE surrogate
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            if training:
                y = (hard - p).detach() + p
            else:
                y = hard

            # soft aggregation
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = token_patches
            
            batch_list.append(token_feats)
            # [新增 4] 将最终确定的索引加入列表
            batch_indices_list.append(kept_idx)

            # fill details
            if return_details:
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)
        # [新增 5] 将当前 batch 的索引列表加入总列表
        selected_indices.append(batch_indices_list)

    # [新增 6] 修改返回值
    if return_details:
        return selected, selected_indices, details
    else:
        return selected, selected_indices

def select_features_network_mmr_variable_k_gpu_purefeature_finetune(
    features, attn_map, query, selector_model, M=100, 
    tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False, k_cur=10,
    return_details=False, device=None,
    # --- soft MMR specific (kept disabled as requested) ---
    enable_soft_mmr=False,  # ← 默认禁用
    beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
):
    """
    Updated to support:
      - Training: returns soft-weighted candidate features (full info + grad)
      - Inference: returns hard-selected original features (efficient)
    MMR remains disabled.
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs, P, d]
    selected = []  # will store list of [num_selected, d] tensors per token
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P, d]
        feats_norm_b = feats_norm[bi]  # [P, d]
        attn_b = attn_map[bi]          # [T, P]
        q_b = query[bi]                # [T, d_q]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # --- Step 1: Build candidate pool (top-M by attention) ---
            idx_candidates = torch.arange(P, device=device)
            order = torch.argsort(scores, descending=True)
            idx_candidates = idx_candidates[order][:M]
            M_actual = idx_candidates.numel()

            if M_actual == 0:
                empty_tensor = torch.zeros(0, d, device=device)
                batch_list.append(empty_tensor)
                if return_details:
                    for key in details:
                        details[key][bi][ti] = torch.zeros(0, device=device) if key != 'soft_agg' else torch.zeros(d, device=device)
                continue

            # --- Step 2: Get candidate features ---
            cand_feats = feats_b[idx_candidates]           # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates] # [M_actual, d]
            cls_prior = scores[idx_candidates]             # [M_actual]

            # --- Step 3: Score with learnable selector ---
            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            logits_mmr = logits_raw  # MMR disabled

            # --- Step 4: Soft probability and top-k ---
            p = torch.softmax(logits_mmr, dim=0)           # [M_actual]
            topk_vals, topk_pos = torch.topk(p, min(k_cur, M_actual), dim=0)
            kept_idx_global = idx_candidates[topk_pos]     # [k_eff]

            # Ensure minimum tokens
            k_eff = kept_idx_global.numel()
            if k_eff < min_token_num:
                need = min_token_num - k_eff
                kept_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if k_eff > 0:
                    local_kept_pos = (idx_candidates.unsqueeze(1) == kept_idx_global.unsqueeze(0)).any(dim=1)
                    kept_mask = local_kept_pos
                remaining = (~kept_mask).nonzero(as_tuple=True)[0]
                if remaining.numel() > 0:
                    add_pos = remaining[:need]
                    kept_idx_global = torch.cat([kept_idx_global, idx_candidates[add_pos]], dim=0)
                    k_eff = kept_idx_global.numel()

            # --- Step 5: STE for surrogate weights ---
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            if training:
                y = (hard - p).detach() + p  # STE: grad flows through p
            else:
                y = hard

            # --- Step 6: Main output ---
            if training:
                # ✅ TRAINING: return soft-weighted full candidate features (retain info + grad)
                # Shape: [M_actual, d]
                output_feats = y.unsqueeze(-1) * cand_feats  # element-wise weighting
            else:
                # ✅ INFERENCE: return hard-selected original features (efficient)
                # Shape: [k_eff, d]
                output_feats = feats_b[kept_idx_global]

            batch_list.append(output_feats)

            # --- Step 7: Fill details if needed ---
            if return_details:
                soft_agg = torch.matmul(y, cand_feats)  # [d]
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx_global
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_network_mmr_variable_k_gpu(
    features,
    attn_map,
    query,
    M=100,
    lambda_mmr=1.0,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1,
    selector_model=None,
    training=False,
    return_details=False,
    device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True,
    beta_mmr=0.8,
    mmr_exclude_self=True,
    mmr_normalize_redundancy=True,
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector + 可微软化 MMR.
    新增 soft-MMR 参数:
    enable_soft_mmr: bool, 是否启用可微软化 MMR（默认 True）
    beta_mmr: float, redundancy 惩罚系数（越大越鼓励多样性）
    mmr_exclude_self: bool, 是否在 S 中置零对角（避免 self contribute）
    mmr_normalize_redundancy: bool, 是否把 redundancy 标准化（/sum(p0) 或 /max）以更稳定调参

    返回与之前一致；若 return_details=True，会在 details 中额外返回 'logits_raw' 和 'logits_mmr' 供分析。
    """
    import torch
    import torch.nn.functional as F

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    # 如果没有外部 selector，则用默认（调用者如果要训练 selector，应在外部实例化并传入该实例）
    if selector_model is None:
        # 轻量默认 selector（与之前示例一致）
        class DefaultSelector(torch.nn.Module):
            def __init__(self, d_feat, d_query=None, proj_dim=256, use_prior=True, tau=0.07):
                super().__init__()
                if d_query is None:
                    d_query = d_feat
                self.wp = torch.nn.Linear(d_feat, proj_dim)
                self.wt = torch.nn.Linear(d_query, proj_dim)
                self.mlp = torch.nn.Sequential(
                    torch.nn.LayerNorm(proj_dim * 2 + (1 if use_prior else 0)),
                    torch.nn.Linear(proj_dim * 2 + (1 if use_prior else 0), proj_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_dim, 1)
                )
                self.use_prior = use_prior
                self.tau = tau

            def forward(self, query, cand_feats, cls_prior=None):
                Mloc = cand_feats.shape[0]
                p = F.normalize(self.wp(cand_feats), dim=-1)
                q = F.normalize(self.wt(query.unsqueeze(0) if query.dim()==1 else query), dim=-1)
                q = q.squeeze(0)
                dot = torch.matmul(p, q) / self.tau
                q_exp = q.unsqueeze(0).expand(Mloc, -1)
                if cls_prior is None:
                    prior = torch.zeros(Mloc, 1, device=cand_feats.device)
                else:
                    prior = cls_prior.unsqueeze(1)
                mlp_in = torch.cat([p, q_exp, prior], dim=-1)
                mlp_logits = self.mlp(mlp_in).squeeze(-1)
                logits = dot + mlp_logits
                return logits

        selector_model = DefaultSelector(d_feat=d, d_query=query.shape[2], proj_dim=min(256, d), use_prior=True, tau=0.07)
        selector_model.to(device)

    selector_model.eval() if not training else selector_model.train()

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选，返回仅 query
                batch_list.append(q_b[ti].unsqueeze(0))
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates]  # descending by construction

            if False:
                # 决定 k_cur
                if method == 'cumulative':
                    cum = torch.cumsum(cand_scores, dim=0)
                    pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                    if pos.numel() == 0:
                        n_req = M_actual
                    else:
                        n_req = int(pos[0].item()) + 1
                    k_cur = max(min_k, min(n_req, max_k, M_actual))
                elif method == 'perplexity':
                    p_dist = cand_scores / (cand_scores.sum() + eps)
                    entropy = -(p_dist * (p_dist + eps).log()).sum()
                    perp = torch.exp(entropy).item()
                    k_cur = int(round(perp))
                    k_cur = max(min_k, min(k_cur, max_k, M_actual))
                elif method == 'attn_mass':
                    mass = cand_scores.sum().item()
                    k_cur = int(round(min_k + (max_k - min_k) * mass))
                    k_cur = max(min_k, min(k_cur, max_k, M_actual))
                else:
                    raise ValueError("Unknown method")

                k_cur = max(k_cur, min_token_num)
                k_cur = min(k_cur, M_actual)
            else:
                k_cur = 10

            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留（不需 selector 排序）
            if M_actual <= k_cur:
                kept_idx = idx_candidates
                if return_details:
                    details['p'][bi][ti] = torch.ones(M_actual, device=device)
                    details['indices'][bi][ti] = kept_idx
                    details['surrogate_weights'][bi][ti] = torch.ones(M_actual, device=device)
                    details['soft_agg'][bi][ti] = torch.matmul(details['surrogate_weights'][bi][ti], feats_b[kept_idx])
                    details['logits_raw'][bi][ti] = torch.zeros(M_actual, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(M_actual, device=device)
                token_patches = feats_b[kept_idx]  # [k_cur, d]
                token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
                batch_list.append(token_feats)
                continue

            # ########################
            # 用 selector 在候选池上打分（可训练 head）
            # ########################
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]  # [M_actual, d] (已归一化)
            cls_prior = cand_scores  # prior

            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            logits_mmr = logits_raw

            if enable_soft_mmr and M_actual > 1:
                # 计算相似矩阵 S (cosine), 排除对角可选
                S = torch.matmul(cand_feats_norm, cand_feats_norm.t())  # [M_actual, M_actual]
                if mmr_exclude_self:
                    # 把对角置0，避免 self contribute
                    try:
                        S = S.clone()
                        S.fill_diagonal_(0.0)
                    except Exception:
                        # 兼容旧 pytorch: 手动减去对角
                        diag = torch.diag(torch.diag(S))
                        S = S - diag

                # base selection probability
                p0 = torch.sigmoid(logits_raw)  # [M_actual]
                # redundancy: S @ p0
                redundancy = torch.matmul(S, p0)  # [M_actual]

                if mmr_normalize_redundancy:
                    # 标准化 redundancy，避免 scale 问题：常见选择是 / (p0.sum()+eps) 或 / redundancy.max()
                    denom = p0.sum().clamp(min=eps)
                    redundancy = redundancy / denom
                    # 也可用 max normalization： redundancy = redundancy / (redundancy.max()+eps)

                # 调整 logits
                logits_mmr = logits_raw - beta_mmr * redundancy

            # logits -> soft scores
            p = torch.sigmoid(logits_mmr)  # [M_actual]

            # top-k on p
            topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)  # positions in candidate pool
            kept_idx = idx_candidates[topk_pos]  # original global indices

            # hard mask & STE surrogate
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            if training:
                y = (hard - p).detach() + p
            else:
                y = hard

            # soft aggregation
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

            # fill details
            if return_details:
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_mmr_variable_k_gpu(features,
    attn_map,
    query,
    M=100,
    lambda_mmr=0.99,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1):
    """
    MMR + variable k per token (GPU-friendly).
    features: [bs, P, d] (torch.Tensor on device)
    attn_map: [bs, T, P]
    query:    [bs, T, d]
    M: candidate pool size (top-M by attention)
    method: 'cumulative' | 'perplexity' | 'attn_mass'
    - cumulative: use alpha (e.g., 0.9) on cumulative attention
    - perplexity: use entropy->perplexity
    - attn_mass: map candidate mass to k
    min_k/max_k: bounds for k per token
    min_token_num: ensure at least this many patches (fallback)
    返回:
    list(bs) -> list(T) -> Tensor [1 + k_cur, d] (query 前置)
    """
    import torch
    import torch.nn.functional as F
    device = features.device
    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                # print(mask.sum())
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()
            #if M_actual == 0:
            #    batch_list.append(q_b[ti].unsqueeze(0))
            #    continue

            # print(idx_candidates)

            cand_scores = scores[idx_candidates]  # descending
            # 决定当前 token 要选多少个 k_cur（不含 query）
            if method == 'cumulative':
                # 找最小 n s.t. cum_sum >= alpha
                cum = torch.cumsum(cand_scores, dim=0)
                pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    n_req = M_actual
                else:
                    n_req = int(pos[0].item()) + 1
                k_cur = max(min_k, min(n_req, max_k, M_actual))
            elif method == 'perplexity':
                p = cand_scores / (cand_scores.sum() + eps)
                entropy = -(p * (p + eps).log()).sum()
                perp = torch.exp(entropy).item()
                k_cur = int(round(perp))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            elif method == 'attn_mass':
                # 候选池内的 attention mass（相对于全部P）
                mass = cand_scores.sum().item()  # in (0,1]
                # 线性映射 mass in [0,1] -> k in [min_k, max_k]
                k_cur = int(round(min_k + (max_k - min_k) * mass))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            else:
                raise ValueError("Unknown method")

            # 保护：至少保留 min_token_num
            k_cur = max(k_cur, min_token_num)
            k_cur = min(k_cur, M_actual)

            # if lambda_mmr == 0:
            #     kept_idx = idx_candidates[-k_cur:]

            # 若 k_cur == 0（理论上不会），直接只返回 query
            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留
            if M_actual <= k_cur:
                kept_idx = idx_candidates
            else:
                # MMR on cand_feats
                cand_feats = feats_norm_b[idx_candidates]  # [M_actual,d]
                sim_mat = torch.matmul(cand_feats, cand_feats.T)  # [M_actual,M_actual]
                selected_positions = []
                selected_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                for step in range(k_cur):
                    if step == 0:
                        _, pick = torch.max(cand_scores, dim=0)
                        pick = int(pick)
                    else:
                        if selected_mask.any():
                            sel_cols = sim_mat[:, selected_mask]  # [M_actual,S]
                            novelty, _ = sel_cols.max(dim=1)
                        else:
                            novelty = torch.zeros(M_actual, device=device)
                        mmr_score = lambda_mmr * cand_scores - (1.0 - lambda_mmr) * (novelty + 1) * 0.5
                        mmr_score = mmr_score.masked_fill(selected_mask, float('-inf'))
                        _, pick = torch.max(mmr_score, dim=0)
                        pick = int(pick)
                    selected_positions.append(pick)
                    selected_mask[pick] = True
                    if selected_mask.all():
                        break
                kept_idx = idx_candidates[torch.tensor(selected_positions, dtype=torch.long, device=device)]

            # print(kept_idx)
            in_ = [l in idx_candidates[:10] for l in kept_idx]
            # print(np.sum(in_))
            # 若仍少于 min_token_num，从余下按 attention 补足
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 拼接 query 在最前
            token_patches = feats_b[kept_idx]  # [k_cur,d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

        selected.append(batch_list)

    return selected

def analyze_image_attention(attention_maps, image_token_start, image_token_end):
    """
    分析每个生成 token 对输入图像 token 的 attention 比重。

    参数:
        attention_maps (list): 每一层的 attention map 列表，每个 attention map 的形状为 [1, 32, 1, k]。
        image_token_start (int): 图像 token 的起始位置。
        image_token_end (int): 图像 token 的结束位置。

    返回:
        dict: 每个生成 token 对图像 token 的 attention 比重，格式为 {token_index: attention_weights}。
    """
    results = {}

    for token_index, layer_attention_maps in enumerate(attention_maps[1:]):
        # 初始化存储每个 token 对图像 token 的 attention 比重
        image_attention_weights = []

        for layer_idx, attention_map in enumerate(layer_attention_maps[:]):
            # attention_map 的形状为 [1, 32, 1, k]
            # 取最后一个维度（k）中对应图像 token 的部分
            image_attention = attention_map[:, :, :, image_token_start:image_token_end].sum(-1) / attention_map[:, :, :, :].sum(-1)

            # 计算每个注意力头对图像 token 的 attention 均值
            # head_attention_weights, _ = image_attention.max(dim=-1)  # 形状为 [1, 32, 1]
            
            head_attention_weights = image_attention
            image_attention_weights.append(head_attention_weights)

        # 将所有层的 attention 权重拼接并计算均值
        image_attention_weights = torch.cat(image_attention_weights, dim=1)  # 形状为 [1, 32 * num_layers, 1]
        image_attention_weights = image_attention_weights.mean(dim=1).squeeze()  # 形状为 [1]

        # 存储结果
        results[token_index] = image_attention_weights.mean().cpu()

    return results

def select_features_mmr_variable_k_gpu(features,
    attn_map,
    query,
    M=50,
    lambda_mmr=0.9,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1):
    """
    MMR + variable k per token (GPU-friendly).
    features: [bs, P, d] (torch.Tensor on device)
    attn_map: [bs, T, P]
    query:    [bs, T, d]
    M: candidate pool size (top-M by attention)
    method: 'cumulative' | 'perplexity' | 'attn_mass'
    - cumulative: use alpha (e.g., 0.9) on cumulative attention
    - perplexity: use entropy->perplexity
    - attn_mass: map candidate mass to k
    min_k/max_k: bounds for k per token
    min_token_num: ensure at least this many patches (fallback)
    返回:
    list(bs) -> list(T) -> Tensor [1 + k_cur, d] (query 前置)
    """
    
    device = features.device
    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                # print(mask.sum())
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()


            cand_scores = scores[idx_candidates]  # descending
            # 决定当前 token 要选多少个 k_cur（不含 query）
            if method == 'cumulative':
                # 找最小 n s.t. cum_sum >= alpha
                cum = torch.cumsum(cand_scores, dim=0)
                pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    n_req = M_actual
                else:
                    n_req = int(pos[0].item()) + 1
                k_cur = max(min_k, min(n_req, max_k, M_actual))
            elif method == 'perplexity':
                p = cand_scores / (cand_scores.sum() + eps)
                entropy = -(p * (p + eps).log()).sum()
                perp = torch.exp(entropy).item()
                k_cur = int(round(perp))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            elif method == 'attn_mass':
                # 候选池内的 attention mass（相对于全部P）
                mass = cand_scores.sum().item()  # in (0,1]
                # 线性映射 mass in [0,1] -> k in [min_k, max_k]
                k_cur = int(round(min_k + (max_k - min_k) * mass))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            else:
                raise ValueError("Unknown method")

            # 保护：至少保留 min_token_num
            k_cur = max(k_cur, min_token_num)
            k_cur = min(k_cur, M_actual)

            # if lambda_mmr == 0:
            #     kept_idx = idx_candidates[-k_cur:]

            # 若 k_cur == 0（理论上不会），直接只返回 query
            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留
            if M_actual <= k_cur:
                kept_idx = idx_candidates
            else:
                # MMR on cand_feats
                cand_feats = feats_norm_b[idx_candidates]  # [M_actual,d]
                sim_mat = torch.matmul(cand_feats, cand_feats.T)  # [M_actual,M_actual]
                selected_positions = []
                selected_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                for step in range(k_cur):
                    if step == 0:
                        _, pick = torch.max(cand_scores, dim=0)
                        pick = int(pick)
                    else:
                        if selected_mask.any():
                            sel_cols = sim_mat[:, selected_mask]  # [M_actual,S]
                            novelty, _ = sel_cols.max(dim=1)
                        else:
                            novelty = torch.zeros(M_actual, device=device)
                        mmr_score = lambda_mmr * cand_scores - (1.0 - lambda_mmr) * (novelty + 1) * 0.5
                        mmr_score = mmr_score.masked_fill(selected_mask, float('-inf'))
                        _, pick = torch.max(mmr_score, dim=0)
                        pick = int(pick)
                    selected_positions.append(pick)
                    selected_mask[pick] = True
                    if selected_mask.all():
                        break
                kept_idx = idx_candidates[torch.tensor(selected_positions, dtype=torch.long, device=device)]

            # 若仍少于 min_token_num，从余下按 attention 补足
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 拼接 query 在最前
            token_patches = feats_b[kept_idx]  # [k_cur,d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

        selected.append(batch_list)

    return selected

def get_retrieval():
    from paths_config import ORGAN_CENTERS_NPY, REGION_REPORT_CSV
    centers = np.load(ORGAN_CENTERS_NPY, allow_pickle=True)

    # 你关心的anatomy顺序（合并mediastinum和heart）
    anatomy_list = [
        "trachea and bronchie", "mediastinum+heart", "lung", "esophagus",
        "pleura", "bone", "thyroid", "breast", "abdomen", "others"
    ]

    # 结果字典
    result = {}

    with open(REGION_REPORT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vol = row['Volumename']
            anatomy = row['Anatomy'].strip().lower()
            sentence = row['Sentence']
            if anatomy in ["mediastinum", "heart"]:
                if vol not in result:
                    # 初始化为10个空字符串
                    result[vol] = [''] * len(anatomy_list)
                # 先把mediastinum和heart都放到同一个位置
                idx = 1  # anatomy_list中"mediastinum+heart"的位置
                if result[vol][idx]:
                    # 已有内容，拼接
                    result[vol][idx] += ' ' + sentence
                else:
                    result[vol][idx] = sentence
            elif anatomy in ["trachea and bronchie", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]:
                if vol not in result:
                    result[vol] = [''] * len(anatomy_list)
                idx = anatomy_list.index(anatomy if anatomy != "mediastinum" and anatomy != "heart" else "mediastinum+heart")
                result[vol][idx] = sentence

    key_reports = [{}]*10
    for idx in range(10):
        for key in list(centers[idx].keys()):
            key_reports[idx][key] = result[key][idx]
    
    return centers, key_reports

def select_features(features, attn_map, b=10, r=0.6):
    """
    features: [bs, 4096, 768]
    attn_map: [bs, 10, 4096]
    b: 每个token选择的特征数
    r: 余弦相似度阈值
    """
    bs, num_tokens, num_feats = attn_map.shape
    feat_dim = features.shape[-1]
    selected_features = []

    for batch_idx in range(bs):
        batch_selected = []
        feats = features[batch_idx]  # [4096, 768]
        attn = attn_map[batch_idx]   # [10, 4096]

        for token_idx in range(num_tokens):
            attn_scores = attn[token_idx]  # [4096]
            # 按分数从大到小排序
            sorted_scores, sorted_indices = torch.sort(attn_scores, descending=True)
            token_selected = []
            for idx in sorted_indices:
                candidate_feat = feats[idx]  # [768]
                # 如果还没选满b个，且与已选的都小于r
                if len(token_selected) == 0:
                    token_selected.append(candidate_feat)
                else:
                    # 计算与已选的余弦相似度
                    selected_feats = torch.stack(token_selected)  # [n, 768]
                    candidate_feat_norm = F.normalize(candidate_feat.unsqueeze(0), dim=1)  # [1, 768]
                    selected_feats_norm = F.normalize(selected_feats, dim=1)  # [n, 768]
                    # 计算与所有已选的最大相似度
                    sim = torch.matmul(selected_feats_norm, candidate_feat_norm.T).squeeze(1)  # [n]
                    max_sim = sim.abs().max()  # 绝对值，防止负相关
                    if max_sim < r:
                        token_selected.append(candidate_feat)
                if len(token_selected) >= b:
                    break
            # 补零
            if len(token_selected) < b:
                num_pad = b - len(token_selected)
                pad_feats = [torch.zeros(feat_dim, device=features.device, dtype=features.dtype) for _ in range(num_pad)]
                token_selected.extend(pad_feats)
            # 选出的b个特征
            token_selected = torch.stack(token_selected)  # [b, 768]
            batch_selected.append(token_selected)
        # 10组特征cat
        batch_selected = torch.cat(batch_selected, dim=0)  # [10*b, 768]
        selected_features.append(batch_selected)
    # [bs, 10*b, 768]
    selected_features = torch.stack(selected_features, dim=0)
    return selected_features

def select_features_minpos(features, attn_map, b=10):
    """
    features: [bs, 4096, 768]
    attn_map: [bs, 10, 4096]
    b: 每个 token 要选出的特征数
    return:   [bs, 10*b, 768]
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    device   = features.device
    out = []

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        attn  = attn_map[bi]          # [10, 4096]
        batch_tokens = []

        for ti in range(num_tokens):
            score = attn[ti]          # [4096]

            # 1) 只保留正分
            pos_mask = score > 0
            if pos_mask.sum() == 0:   # 无正分
                batch_tokens.append(torch.zeros(b, feat_dim, device=device))
                continue

            pos_idx   = torch.where(pos_mask)[0]
            pos_score = score[pos_idx]

            # 2) 按正分升序取前 b 个
            _, order = torch.sort(pos_score)
            chosen_idx = pos_idx[order[:b]]        # 最多 b 个
            token_feats = feats[chosen_idx]        # [min(b,N), 768]

            # 3) 不足 b 时零填充
            pad_len = b - token_feats.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, feat_dim, device=device)
                token_feats = torch.cat([token_feats, pad], dim=0)

            batch_tokens.append(token_feats)

        out.append(torch.cat(batch_tokens, dim=0))   # [10*b, 768]

    return torch.stack(out, dim=0)    

def select_features_segment(features, attn_map, b=10, eps=1e-8):
    """
    features  : [bs, 4096, 768]
    attn_map  : [bs, 10, 4096]
    b         : 每个 token 要选出的特征数
    eps       : 防止除以 0
    return    : [bs, 10*b, 768]   与原函数完全一致
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    selected = []

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        attn  = attn_map[bi]          # [10, 4096]
        batch_tokens = []

        for ti in range(num_tokens):
            score = attn[ti]          # [4096]

            # 1) 只保留正分
            pos_mask = score > 0
            if pos_mask.sum() == 0:     # 无正分，直接全零
                batch_tokens.append(torch.zeros(b, feat_dim, device=features.device))
                continue

            pos_idx   = torch.where(pos_mask)[0]
            pos_score = score[pos_idx]  # [N_pos]

            # 2) 把正分区间切成 b 段
            min_s, max_s = pos_score.min(), pos_score.max()
            if max_s - min_s < eps:     # 所有正分相同，直接选前 b 个
                chosen_idx = pos_idx[:b]
            else:
                seg_width = (max_s - min_s) / b
                chosen_idx = []
                for k in range(b):
                    lower = min_s + k * seg_width
                    upper = min_s + (k + 1) * seg_width
                    seg_mask = (pos_score >= lower) & (pos_score < upper)
                    # 段内取最高
                    if seg_mask.any():
                        seg_idx = pos_idx[seg_mask][pos_score[seg_mask].argmax()]
                    else:
                        seg_idx = pos_idx[0]        # fallback：选全局第一个
                    chosen_idx.append(seg_idx)

            chosen_idx = torch.tensor(chosen_idx, device=features.device)
            token_feats = feats[chosen_idx]          # [b, 768]

            # 3) 若不足 b 个，零填充
            if token_feats.shape[0] < b:
                pad = torch.zeros(b - token_feats.shape[0], feat_dim,
                                  device=features.device)
                token_feats = torch.cat([token_feats, pad], dim=0)

            batch_tokens.append(token_feats)

        # 10 组拼接
        selected.append(torch.cat(batch_tokens, dim=0))   # [10*b, 768]

    return torch.stack(selected, dim=0)      

def select_features_top100_segment(features, attn_map, b=10, r=0.95):
    """
    features: [bs, 4096, 768]
    attn_map: [bs, 10, 4096]
    b: 每个 token 要选出的特征数
    r: 余弦相似度阈值（去重用）
    return:   [bs, 10*b, 768]
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    device = features.device
    out = []

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        attn  = attn_map[bi]          # [10, 4096]
        batch_tokens = []

        for ti in range(num_tokens):
            score = attn[ti]          # [4096]

            # 1) 取前 100 高分
            top_scores, top_idx = torch.topk(score, k=100, largest=True)
            top_feats = feats[top_idx]            # [100, 768]



            # 2) 将 100 个分数切成 b 段（等宽）
            if b == 1:        # 避免除 0
                seg_width = 1.0
            else:
                seg_width = (top_scores.max() - top_scores.min()) / b

            chosen = []
            for k in range(b):
                low  = top_scores.min() + k * seg_width
                high = top_scores.min() + (k + 1) * seg_width
                # 段内候选
                mask = (top_scores >= low) & (top_scores < high)
                if not mask.any():                  # 空段
                    mask = (top_scores == top_scores.min())  # fallback 到最低分
                seg_idx = top_idx[mask][top_scores[mask].argmax()]   # 段内最高分
                seg_feat = feats[seg_idx]

                # 3) 余弦去重
                if len(chosen) == 0:
                    chosen.append(seg_feat)
                else:
                    cand_norm = F.normalize(seg_feat.unsqueeze(0), dim=1)
                    sel_norm  = F.normalize(torch.stack(chosen), dim=1)
                    max_sim   = torch.matmul(sel_norm, cand_norm.T).abs().max()
                    if max_sim < r:
                        chosen.append(seg_feat)

                if len(chosen) >= b:
                    break

            # 4) 补齐 b 个
            chosen = torch.stack(chosen) if chosen else torch.empty(0, feat_dim, device=device)
            pad_len = b - chosen.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, feat_dim, device=device)
                chosen = torch.cat([chosen, pad], dim=0)

            batch_tokens.append(chosen)             # [b, 768]

        out.append(torch.cat(batch_tokens, dim=0))  # [10*b, 768]

    return torch.stack(out, dim=0)         

def select_features_weighted_cover(features, attn_map, b=10):
    """
    features : [bs, 4096, 768]
    attn_map : [bs, 10, 4096]  每个 token 的 CLS-token attention 分数
    b        : 每个 token 要选出的特征数
    return   : [bs, 10*b, 768]
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    device   = features.device
    out = []

    # 预先归一化特征，方便后续余弦距离
    feats_norm = F.normalize(features, dim=2)  # [bs, 4096, 768]

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        scores = attn_map[bi]         # [10, 4096]

        batch_tokens = []
        for ti in range(num_tokens):
            score = scores[ti]        # [4096]
            token_feats = feats       # [4096, 768]
            token_norm  = feats_norm[bi]

            # 归一化 attention → 权重
            weight = score / (score.sum() + 1e-8)   # [4096]

            # 贪心加权覆盖
            selected_idx = []
            mask = torch.ones_like(weight, dtype=torch.bool)

            # 预先计算余弦距离矩阵（4096×4096，GPU 可承受）
            dist = 1 - torch.matmul(token_norm, token_norm.T)  # [4096, 4096]

            for _ in range(b):
                if not mask.any():
                    break
                if not selected_idx:                       # 第一轮：权重最大
                    idx = torch.argmax(weight * mask.float())
                else:
                    sel = torch.tensor(selected_idx, device=device)
                    # 到已选集合的最小距离
                    min_dist = dist[:, sel].min(dim=1)[0]
                    # 加权后最远
                    idx = torch.argmax(weight * mask.float() * min_dist)
                selected_idx.append(idx)
                mask[idx] = False

            # 若不足 b，用零填充
            if len(selected_idx) < b:
                pad = torch.zeros(b - len(selected_idx), feat_dim, device=device)
                token_out = torch.cat([token_feats[selected_idx], pad], dim=0)
            else:
                idx_tensor = torch.tensor(selected_idx, device=device)[:b]
                if idx_tensor.numel() == 0:
                    token_out = torch.zeros(b, feat_dim, device=device)
                else:
                    token_out = token_feats[idx_tensor]

            batch_tokens.append(token_out)  # [b, 768]

        out.append(torch.cat(batch_tokens, dim=0))  # [10*b, 768]

    return torch.stack(out, dim=0)      # [bs, 10*b, 768]

def select_features_adaptive(features,
                             attn_map,
                             query,
                             r=0.6,
                             tau=0.000,
                             max_per_token=30,
                             min_token_num=1):
    """
    静态、非端到端地根据 attention 分布为每个 token 选 patch。

    新增:
      query: [bs, num_tokens, 768] —— 每个 token 的 query 向量，会插到对应 list 最前面
      min_token_num: 每个 token 最终最少保留的 patch 数（不含 query）

    返回
      list[list[Tensor]]
        外层长度 = batch_size
        内层长度 = num_tokens
        最里层 Tensor 形状 = [1 + k_i, 768]，其中第 0 维是 query, k_i ≥ min_token_num
    """
    bs, num_tokens, _ = attn_map.shape
    selected = []

    for bi in range(bs):
        feats      = features[bi]                # [4096, 768]
        feats_norm = F.normalize(feats, dim=1)
        attn       = attn_map[bi]                # [num_tokens, 4096]
        q_batch    = query[bi]                   # [num_tokens, 768]

        batch_list = []
        for ti in range(num_tokens):
            q_vec   = q_batch[ti]                # [768]
            scores  = attn[ti]                   # [4096]

            # 1. 绝对阈值过滤
            valid_mask = scores >= tau
            idx_above  = valid_mask.nonzero(as_tuple=False).squeeze(-1)

            # print(idx_above)
            # 若阈值后不足 min_token_num，则取全局 top-min_token_num
            if idx_above.numel() < min_token_num:
                idx_above = scores.topk(min_token_num).indices

            # 2. 按 attention 排序 + 硬上限
            scores_above = scores[idx_above]
            _, order = torch.sort(scores_above, descending=True)
            idx_sorted = idx_above[order][:max_per_token]
            # 3. 贪婪去重
            keep_idx, keep_norm = [], []
            for idx in idx_sorted:
                cand_norm = feats_norm[idx]
                if keep_norm:
                    sim = torch.matmul(torch.stack(keep_norm), cand_norm).max()
                    if sim >= r:
                        continue
                keep_norm.append(cand_norm)
                keep_idx.append(idx)

            # 若仍不足 min_token_num，再从剩余高分里补
            if len(keep_idx) < min_token_num:
                need = min_token_num - len(keep_idx)
                remain = [i for i in idx_sorted if i not in keep_idx]
                # 按 attention 从高到低补
                remain_sorted = sorted(remain, key=lambda i: scores[i].item(), reverse=True)
                keep_idx.extend(remain_sorted[:need])

            token_feats = torch.cat([feats[ii:ii+1] for ii in keep_idx], 0)           # [k, 768]
            # print(token_feats.shape, keep_idx)
            token_feats = torch.cat([q_vec.unsqueeze(0), token_feats], dim=0)  # [1+k, 768]
            batch_list.append(token_feats)

        selected.append(batch_list)
    return selected


class QFormer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(QFormer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Define linear projection layers for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multihead attention layers for cross-attention
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward_(self, query, key, value, add_residual=True):
        # Apply linear projections
        query = self.query_proj(query).permute(1, 0, 2)
        key = self.key_proj(key).permute(1, 0, 2)
        value = self.value_proj(value).permute(1, 0, 2)
        
        # Compute first cross-attention
        attn_output1, _ = self.cross_attn1(query, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output1 = attn_output1 + query
        
        # Apply normalization and dropout
        attn_output1 = self.norm1(attn_output1)
        attn_output1 = self.dropout1(attn_output1)
        
        # Compute second cross-attention
        attn_output2, _ = self.cross_attn2(attn_output1, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output2 = attn_output2 + attn_output1
        
        # Apply normalization and dropout
        attn_output2 = self.norm2(attn_output2)
        attn_output2 = self.dropout2(attn_output2)
        
        return attn_output2.permute(1, 0, 2), attn_output2.permute(1, 0, 2)

    def forward__(self, text, img_fea, add_residual=True):
        # Apply linear projections
        query = self.query_proj(text).permute(1, 0, 2)
        key = self.key_proj(img_fea).permute(1, 0, 2)
        value = self.value_proj(img_fea).permute(1, 0, 2)
        
        # Compute first cross-attention
        attn_output1, _ = self.cross_attn1(query, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output1 = attn_output1 + query
        
        # Apply normalization and dropout
        attn_output1 = self.norm1(attn_output1)
        attn_output1 = self.dropout1(attn_output1)
        
        # Compute second cross-attention and get attention weights
        attn_output2, attn_weights = self.cross_attn2(attn_output1, key, value, need_weights=True)
        
        # Optionally add residual connection
        if add_residual:
            attn_output2 = attn_output2 + attn_output1
        
        # Apply normalization and dropout
        attn_output2 = self.norm2(attn_output2)
        attn_output2 = self.dropout2(attn_output2)
        
        # Average attention weights across heads
        avg_attn_weights = attn_weights.mean(dim=0)  # Shape: (query_len, key_len)
        
        # Select top 10 keys for each query
        top_k_values = []
        for i in range(avg_attn_weights.size(0)):  # Iterate over each query
            top_k_indices = avg_attn_weights[i].topk(10, dim=-1).indices
            selected_values = value[top_k_indices, :, :]  # Select top 10 values
            top_k_values.append(selected_values)
        
        # Concatenate selected values
        top_k_values = torch.cat(top_k_values, dim=0)  # Shape: (10 * query_len, batch_size, embed_dim)
        
        # Permute top_k_values to match the output shape
        top_k_values = top_k_values.permute(1, 0, 2)
        
        # Concatenate the first output with the selected top_k_values
        combined_output = torch.cat((attn_output2.permute(1, 0, 2), top_k_values), dim=1)
        
        return attn_output2.permute(1, 0, 2), combined_output

    def forward(self, query, img_fea, add_residual=True,
            dominant_num=10, contextual_num=0):

        """
        query   : [B, Lq, D]   text tokens
        img_fea : [B, Li, D]   image tokens
        """
        """
        query   : [B, Lq, D]   text tokens
        img_fea : [B, Li, D]   image tokens
        """
        B, Lq, D = query.shape
        _, Li, _ = img_fea.shape

        # 1. 原两层 cross-attention，拿 attn_weights
        q = self.query_proj(query).permute(1, 0, 2)      # [Lq, B, D]
        k = self.key_proj(img_fea).permute(1, 0, 2)      # [Li, B, D]
        v = self.value_proj(img_fea).permute(1, 0, 2)    # [Li, B, D]

        attn_out1, _ = self.cross_attn1(q, k, v)
        if add_residual:
            attn_out1 = attn_out1 + q
        attn_out1 = self.norm1(attn_out1)
        attn_out1 = self.dropout1(attn_out1)

        attn_out2, attn_weights = self.cross_attn2(
            attn_out1, k, v, need_weights=True)          # attn_weights: [B, Lq, Li]

        if add_residual:
            attn_out2 = attn_out2 + attn_out1
        attn_out2 = self.norm2(attn_out2)
        attn_out2 = self.dropout2(attn_out2)

        text_out = attn_out2.permute(1, 0, 2)            # [B, Lq, D]  保持原接口

        # 2. 每个 image token 的归属：被哪个 text token 最大关注
        owner = attn_weights.argmax(dim=1)               # [B, Li] 取值 0..Lq-1

        # 3. 逐 text token 做 VisionZIP
        #    先准备好 key/value（key 直接当 metric）
        k = k.permute(1, 0, 2)    # [B, Li, D]
        v = v.permute(1, 0, 2)    # [B, Li, D]

        # 用于收集最终结果
        zipped_list = []

        for b in range(B):
            # 取出当前 batch 数据
            k_b = k[b]             # [Li, D]
            v_b = v[b]             # [Li, D]
            owner_b = owner[b]     # [Li]

            zipped_per_text = []
            for t in range(Lq):
                # 属于该 text token 的 image token 索引
                idx = torch.where(owner_b == t)[0]
                if idx.numel() == 0:          # 没有归属的 img token，用空 tensor
                    # 可以给一个可学习的 dummy token，或跳过
                    # 这里简单用 0 填充
                    comp = torch.zeros(dominant_num + contextual_num, D,
                                    device=k.device, dtype=k.dtype)
                    zipped_per_text.append(comp)
                    continue

                group_k = k_b[idx]           # [N_group, D]
                group_v = v_b[idx]

                N_group = group_k.size(0)
                if N_group <= dominant_num + contextual_num:
                    # token 太少，全部保留再 pad
                    pad = dominant_num + contextual_num - N_group
                    comp = torch.cat([group_v,
                                    torch.zeros(pad, D, device=k.device)], dim=0)
                    zipped_per_text.append(comp)
                    continue

                # 计算每个 img token 对该 text token 的注意力权重
                w = attn_weights[b, t, idx]  # [N_group]

                # 3.1 dominant
                _, dom_idx = torch.topk(w, dominant_num)
                dom_k = group_k[dom_idx]     # [dominant_num, D]
                dom_v = group_v[dom_idx]

                if contextual_num == 0:
                    comp = dom_v          # 只保留 dominant
                else:
                    # 3.2 contextual
                    mask = torch.ones(N_group, dtype=torch.bool, device=k.device)
                    mask[dom_idx] = False
                    remain_k = group_k[mask]
                    remain_v = group_v[mask]

                    step = max(1, remain_k.size(0) // contextual_num)
                    tgt_idx = torch.arange(0, remain_k.size(0), step)[:contextual_num].to(k.device)
                    tgt_k = remain_k[tgt_idx]
                    tgt_v = remain_v[tgt_idx]
                    # print(remain_k.device, tgt_idx.device)
                    merge_k = remain_k[~torch.isin(torch.arange(remain_k.size(0), device=k.device), tgt_idx)]
                    merge_v = remain_v[~torch.isin(torch.arange(remain_v.size(0), device=k.device), tgt_idx)]

                    sim = torch.mm(merge_k, tgt_k.t())                   # [N_merge, contextual_num]
                    assign = sim.argmax(dim=1)
                    one_hot = torch.zeros_like(sim).scatter_(1, assign.unsqueeze(1), 1)
                    counts = one_hot.sum(dim=0, keepdim=True).clamp(min=1)

                    agg_v = torch.mm(one_hot.t(), merge_v) / counts.t()  # [contextual_num, D]
                    ctx_v = tgt_v + agg_v

                    comp = torch.cat([dom_v, ctx_v], dim=0)              # [N, D]
                zipped_per_text.append(comp)

            # 把 Lq 个结果叠成 [Lq, N, D] 再转置回 [B, Lq, N, D]
            zipped_tensor = torch.stack(zipped_per_text, dim=0)      # [Lq, N, D]
            zipped_list.append(zipped_tensor)

        # 最终形状 [B, Lq, N, D] ；若下游只关心图级别，可做 pooling
        zipped_img = torch.stack(zipped_list, dim=0)               # [B, Lq, N, D]

        return text_out, torch.cat([text_out, zipped_img.reshape((text_out.shape[0], -1, zipped_img.shape[-1]))], 1)
