
import json, os, shutil, re, random, io, time
import torch
from scipy.special import expit
import sys
# Make the repo root importable (classifier.py and paths_config live there).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier import RadBertClassifier
from paths_config import RADBERT_PATH, RADBERT_CLASSIFIER_CKPT, GRPO_F1_SERVER_PORT, require
import uuid
from bottle import request, Bottle


def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)
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

if __name__ == '__main__':   
    require("RADBERT_PATH", "RADBERT_CLASSIFIER_CKPT")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    ####for the ce metric cal
    tokenizer_ce = AutoTokenizer.from_pretrained(RADBERT_PATH, do_lower_case=True)
    classificer = RadBertClassifier(n_classes=18).cuda()
    classificer.load_state_dict(torch.load(RADBERT_CLASSIFIER_CKPT, map_location="cpu"))
    classificer.eval()
    classificer.requires_grad_(False)

    def get_F1_score(inputs):
        print(inputs)

        encodings = tokenizer_ce(inputs, return_tensors='pt',max_length=512,padding='max_length',truncation=True)
        
        input_ids = encodings['input_ids'].cuda()
        attention_mask = encodings['attention_mask'].cuda()

        pred_logits = classificer(input_ids, attention_mask).cpu().numpy()# print(pred_logits.shape)
        # pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        
        pred_labels[pred_labels>=0.5]=1
        pred_labels[pred_labels<0.5]=0
        return pred_labels

    # 使用队列来处理请求
    raw_queue = queue.Queue()
    result_dict = {}

    app = Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        request_id = str(uuid.uuid4())  # 生成唯一请求ID
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        data = {'base': json.loads(dd[0]), 'inputs': json.loads(dd[1].decode('utf-8')), 'id': request_id}
        raw_queue.put(data)
        print('receive', data['inputs'])
        return request_id.encode('utf-8')  # 返回请求ID给客户端

    @app.route('/get/<request_id>', method='GET')
    def do_get(request_id):
        if request_id not in result_dict:
            return b'empty'
        result = result_dict.pop(request_id)  # 获取并移除结果
        return result

    def run_server():
        bottle.run(app, host='0.0.0.0', port=GRPO_F1_SERVER_PORT, server='tornado')

    threading.Thread(target=run_server, daemon=False).start()

    while True:
        d = raw_queue.get()
        request_id = d['id']
        with torch.inference_mode():
            pred_results = get_F1_score(d['inputs'])
        data = [tensor_to_bytes(torch.tensor(pred_results))]
        print(pred_results)
        xdata = make_bytes_list(data)
        result_dict[request_id] = xdata  # 将结果存储到字典中

    