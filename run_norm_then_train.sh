export HF_LEROBOT_HOME=/mnt/public/gaojiahuan/huggingface/lerobot
export HF_DATASETS_OFFLINE=1

export PATH=/usr/local/cuda/bin:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# 归一化
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run scripts/compute_norm_stats.py --config-name restock_multi_goods_sm2sm

export WANDB_PROJECT="openpi"
export WANDB_ENTITY="striding-algo"

# 训练（还可以设定 WANDB_API_KEY参数）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py restock_multi_goods_sm2sm --overwrite --data.random-drop-master 0.10 --data.random-drop-history 0.50 --data.random-drop-future 0.50 --data.random-pos-offset 0.020 