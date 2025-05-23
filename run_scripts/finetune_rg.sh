num_gpus=1
per_gpu_batchsize=1
# kill -STOP 62
# -X faulthandler
# main_report_gen_rate
# main_report_gen_dpo
# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=2 python3 main_report_gen_rate.py \
 with data_root=data/pretrain_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_rg \
 per_gpu_batchsize=${per_gpu_batchsize} 
 resume_from=/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/result/task_pretrain_m3ae-seed0-from_/version_2157/checkpoints/last.ckpt
# kill -CONT 62