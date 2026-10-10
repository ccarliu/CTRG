import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 竞争稀疏 cross-attention ----------
class CompetingSparseCrossAttn(nn.Module):
    def __init__(self, dim, num_heads=8, k_sparse=16, init_tau=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.k_sparse  = k_sparse
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.tau       = nn.Parameter(torch.tensor(float(init_tau)))

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, cls_tokens, patch_tokens):
        B, N, D = cls_tokens.shape
        L = patch_tokens.shape[1]
        H = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(cls_tokens).view(B, N, H, Dh).transpose(1, 2)  # [B,H,N,Dh]
        k = self.k_proj(patch_tokens).view(B, L, H, Dh).transpose(1, 2)  # [B,H,L,Dh]
        v = self.v_proj(patch_tokens).view(B, L, H, Dh).transpose(1, 2)  # [B,H,L,Dh]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale      # [B,H,N,L]

        comp_weights = F.softmax(logits / torch.clamp(self.tau.abs(), min=1e-4), dim=2)
        topk_val, topk_idx = torch.topk(comp_weights, k=min(self.k_sparse, L), dim=-1)

        sparse_attn = torch.zeros_like(comp_weights)
        sparse_attn.scatter_(-1, topk_idx, topk_val)
        sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(sparse_attn, v)          # [B,H,N,Dh]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, sparse_attn            # 额外返回 attention map

# ---------- 一层 Decoder ----------
class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, k_sparse, init_tau, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CompetingSparseCrossAttn(dim, num_heads, k_sparse, init_tau)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, cls_tokens, patch_tokens):
        # 自注意力
        cls_tokens = cls_tokens + self.self_attn(
            self.norm1(cls_tokens), self.norm1(cls_tokens), self.norm1(cls_tokens))[0]

        # 竞争稀疏 cross-attention
        cls_tokens, attn = self.cross_attn(self.norm2(cls_tokens), patch_tokens)

        # FFN
        cls_tokens = cls_tokens + self.mlp(self.norm3(cls_tokens))
        return cls_tokens, attn

# ---------- 多层 CompetingQFormer ----------
class CompetingQFormer(nn.Module):
    def __init__(self, dim=768, num_heads=12, k_sparse=16, init_tau=1.0,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim, num_heads, k_sparse, init_tau, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, cls_tokens, patch_tokens):
        attn_map = None
        for layer in self.layers:
            cls_tokens, attn = layer(cls_tokens, patch_tokens)
            attn_map = attn      # 只保留最后一层
        return cls_tokens, attn_map   # [B, H, N, L]