num_gpus=3
per_gpu_batchsize=1
#kill -STOP 61
# -X faulthandler
CUDA_VISIBLE_DEVICES=1,2,3 python3 -X faulthandler main_3D.py \
 with data_root=data/pretrain_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_pretrain_m3ae_3D \
 per_gpu_batchsize=${per_gpu_batchsize} 
 # resume_from=/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/result/task_pretrain_m3ae-seed0-from_/version_2263/checkpoints/last.ckpt
#kill -CONT 61
# 79
 