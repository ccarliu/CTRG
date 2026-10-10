import json, os, shutil, re, random, io, time
import sys
import torch
from peft import get_peft_model, LoraConfig, TaskType
import uuid

# Make the repo root importable so paths_config resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import LLM_DECODER_PATH, GRPO_INIT_CKPT, GRPO_REF_SERVER_PORT, require

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
    require("LLM_DECODER_PATH", "GRPO_INIT_CKPT")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_path = LLM_DECODER_PATH # change the model here
    # model_path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-3.2-1B"

    ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="cuda")

    lora_config = LoraConfig(
                r=64, # 64
                lora_alpha=16,
                target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj",],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
                #modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=16, lora_dropout=0.05,target_modules=["W_pack", "o_proj"]
    )

    ref_model = get_peft_model(ref_model, peft_config)
    ref_model.print_trainable_parameters()
    print('Loading LLAMA LoRA Done') 

    # Reference policy = the initial D_pro (same checkpoint the GRPO client starts from).
    ckpath = GRPO_INIT_CKPT
    if not (ckpath and os.path.exists(ckpath)):
        raise FileNotFoundError(f"GRPO_INIT_CKPT not found: '{ckpath}' (Stage-3 D_pro checkpoint)")

    ck = torch.load(ckpath)["state_dict"]
    nck = {k.replace("text_decoder.", ""): v for k, v in ck.items() if "text_decoder" in k}
    
    ref_model.load_state_dict(nck)
    print("load pretrained weights from: ", ckpath)

    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_embedding, prompt_len, answer_ids):
        logits = ref_model(inputs_embeds = input_embedding.cuda()).logits  # (B, L, V)

        # print(logits.shape, answer_ids.shape, prompt_len)
        logits = logits[:, prompt_len:-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = answer_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)

        # print(torch.stack(per_token_logps).cpu().shape)
        return torch.stack(per_token_logps).cpu()

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    result_dict = {}

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        request_id = str(uuid.uuid4())  # 生成唯一请求ID
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
 
        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        data['answer_ids'] = bytes_to_tensor(dd[3])
        data['id'] = request_id
        data['gen_logps'] = bytes_to_tensor(dd[4])

        raw_queue.put(data)
        # print('receive', data['inputs'].shape, data['rewards'], data['gen_logps'].shape if 'gen_logps' in data else '')
        return request_id.encode('utf-8')

    @app.route('/get/<request_id>', method='GET')
    def do_get(request_id):
        if request_id not in result_dict:
            return b'empty'
        result = result_dict.pop(request_id)  # 获取并移除结果
        return result
    
    def run_server(): bottle.run(app, host='0.0.0.0', port=GRPO_REF_SERVER_PORT, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()
    ii = 0
    while True:
        d = raw_queue.get()
        request_id = d['id']
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device), prompt_length, d['answer_ids'].to(ref_model.device))

        ###############################
        ii += 1

        # per_token_logps = per_token_logps[:,prompt_length-1:]
        data = [json.dumps(d['base']).encode(), 
                tensor_to_bytes(d['rewards']), tensor_to_bytes(per_token_logps)]
        if 'gen_logps' in d: data.append(tensor_to_bytes(d['gen_logps']))
        xdata = make_bytes_list(data)
        result_dict[request_id] = xdata

    