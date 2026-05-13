export HF_LEROBOT_HOME=/mnt/public/gaojiahuan/huggingface/lerobot
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# 不要把密钥写进仓库。运行前在 shell 里 export WANDB_API_KEY=... 即可。

export PATH=/usr/local/cuda/bin:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda


CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/compute_norm_stats.py --config-name restock_goods_beijing_chengdu_sm2sm

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py restock_goods_beijing_chengdu_sm2sm --overwrite --data.random-drop-master 0.10 --data.random-drop-history 0.50 --data.random-drop-future 0.50 --data.random-pos-offset 0.020 
