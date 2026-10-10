# green_server_simple_fixed.py
import os
import io
import json
import uuid
import time
import sys
import torch
import multiprocessing as mp
import threading
from bottle import Bottle, request, run
from datasets import Dataset
from green_score import GREEN, GreenSingle  # 你的原始 GREEN 類

# Make the repo root importable so paths_config resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import GREEN_MODEL_PATH, GRPO_GREEN_SERVER_PORT, require

# ---------- HTTP / 全局  ----------
app = Bottle()
result_dict = {}     # 存放处理结果（主进程）
main_q = None        # will be set in __main__
worker_qs = []       # will be created in __main__
result_q = None      # will be created in __main__

# ---------- 工具函数 ----------
def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    # torch.load 不接受 weights_only 參數
    return torch.load(io.BytesIO(b), map_location='cpu')

def pack_bytes_list(byte_elms):
    """len+bytes..."""
    buf = io.BytesIO()
    buf.write(len(byte_elms).to_bytes(4, 'big'))
    for x in byte_elms:
        buf.write(len(x).to_bytes(4, 'big'))
        buf.write(x)
    return buf.getvalue()

def unpack_bytes_list(b):
    buf = io.BytesIO(b)
    n = int.from_bytes(buf.read(4), 'big')
    lst = []
    for _ in range(n):
        l = int.from_bytes(buf.read(4), 'big')
        lst.append(buf.read(l))
    return lst
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

# ---------- HTTP 路由（单一定义） ----------
@app.post('/upload')
def do_upload():
    global main_q
    if main_q is None:
        # 服务尚未初始化
        return b'server_not_ready'
    try:
        blist = unpack_bytes_list(request.body.read())
        # 假定 blist[1] 是 inputs，blist[2] 是 gts（与你原始相同）
        data = {
            'inputs': json.loads(blist[1].decode('utf-8')),
            'gts'   : json.loads(blist[2].decode('utf-8')),
            'id'    : str(uuid.uuid4())
        }
    except Exception as e:
        return f'bad_request: {e}'.encode()
    main_q.put(data)
    return data['id'].encode()

@app.get('/get/<req_id>')
def do_get(req_id):
    # 返回二进制打包数据或 'empty'
    return result_dict.pop(req_id, b'empty')

def http_server():
    run(app, host='0.0.0.0', port=GRPO_GREEN_SERVER_PORT, server='tornado')


# ---------------- Worker ----------------
def worker(rank, q_in, q_out):
    # 每个 worker 绑定到一个物理 GPU（通过 CUDA_VISIBLE_DEVICES）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank+1)
    device = torch.device("cuda:0")  # 因为 CUDA_VISIBLE_DEVICES 已经映射
    try:
        scorer = GreenSingle(model_name=GREEN_MODEL_PATH,
                             gpu_id=0, cpu=False)
        scorer.model.eval()
    except Exception as e:
        print(f"[Worker {rank}] failed to init scorer: {e}")
        q_out.put(("__init_fail__", rank, str(e)))
        return

    print(f"[Worker {rank}] ready on GPU {rank}")

    while True:
        job = q_in.get()
        if job is None:
            print(f"[Worker {rank}] received shutdown")
            break
        try:
            # job = (req_id, start_idx, preds_list, gts_list)
            req_id, start_idx, preds, gts = job
            if len(preds) == 0:
                q_out.put((req_id, start_idx, []))
                continue

            ds = Dataset.from_dict({"reference": gts, "prediction": preds})
            ds = scorer.process_data(ds)   # 假定返回的 ds 是子集
            scorer.dataset = ds
            scores = scorer.infer()        # should return list aligned to ds order

            # 返回时带上该片段的起始索引，便于主进程合并并保持顺序
            q_out.put((req_id, start_idx, scores))
            print(f"[Worker {rank}] done {len(scores)} samples (start {start_idx})")
        except Exception as e:
            print(f"[Worker {rank}] exception: {e}")
            q_out.put((req_id, start_idx, []))


# ---------------- 主進程調度 ----------------
def main_loop(num_workers):
    """一个请求 → 拆 N 份 → 分别发 → 收集并按位置合并 → 存结果"""
    global result_dict
    while True:
        d = main_q.get()
        if d is None:
            # 可以用来触发关闭
            break
        req_id, preds, gts = d['id'], d['inputs'], d['gts']
        n = len(preds)
        print("totally number: ", n)
        if n != len(gts):
            # 简单检查
            result_dict[req_id] = pack_bytes_list([json.dumps({"error":"len_mismatch"}).encode()])
            continue

        # 切片成 num_workers 份（按连续块）
        per = (n + num_workers - 1) // num_workers
        jobs = []
        for i in range(num_workers):
            start = i * per
            end = min(start + per, n)
            sub_preds = preds[start:end]
            sub_gts = gts[start:end]
            jobs.append((req_id, start, sub_preds, sub_gts))

        # 将作业发给 worker
        for i in range(num_workers):
            worker_qs[i].put(jobs[i])

        # 收集结果并按位置合并
        final_scores = [None] * n
        received = 0
        while received < num_workers:
            item = result_q.get()
            # 期望 item = (req_id, start_idx, scores)
            if len(item) == 3:
                rid, start_idx, scores = item
                if rid != req_id:
                    # 不同请求的混入（理论上不会发生），可忽略或处理
                    print(f"[Main] warning: got result for {rid} while waiting for {req_id}")
                if start_idx < 0:
                    # error marker
                    received += 1
                    continue
                # fill final_scores
                for i, s in enumerate(scores):
                    pos = start_idx + i
                    if pos < n:
                        final_scores[pos] = s
                received += 1
            else:
                # 兼容早期错误项
                received += 1

        # 检查是否有缺失（None），并用占位符替换
        for i in range(n):
            if final_scores[i] is None:
                final_scores[i] = {"error": "missing_score"}
        print(torch.tensor(final_scores))
        # 打包并存入 result_dict，客户端可通过 /get/<id> 获取
        result_dict[req_id] = make_bytes_list([tensor_to_bytes(torch.tensor(final_scores))])
        print(f"[Main] request {req_id} finished, total {len(final_scores)} scores")


# ---------------- 启动 ----------------
if __name__ == "__main__":
    require("GREEN_MODEL_PATH")
    NUM_WORKERS = 3
    mp.set_start_method("spawn", force=True)

    # 初始化队列（在主进程）
    main_q = mp.Queue()               # HTTP -> 主调度线程
    result_q = mp.Queue()             # Worker -> 主
    worker_qs = [mp.Queue() for _ in range(NUM_WORKERS)]  # 主 -> 各 worker

    # 启动 worker 进程（不要设置 daemon=True，方便管理）
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(i, worker_qs[i], result_q))
        p.start()
        workers.append(p)

    # 启动主调度线程（在主进程内部运行，负责分发/合并）
    threading.Thread(target=main_loop, args=(NUM_WORKERS,), daemon=True).start()

    # 启动 HTTP 服务（阻塞）
    http_server()

    # 可在需要时清理：发送 None 到队列以结束
    # for q in worker_qs: q.put(None)
    # main_q.put(None)
    # for p in workers: p.join()