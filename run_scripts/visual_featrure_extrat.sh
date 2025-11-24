CUDA_VISIBLE_DEVICES=1 python3 pretrained_visual_feature_extract.py \
 with data_root=data/pretrain_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_pretrain_m3ae_3D \
 per_gpu_batchsize=${per_gpu_batchsize}