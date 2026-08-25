set -e
export HF_LEROBOT_HOME=/mnt/public/gaojiahuan/huggingface/lerobot
export HF_HOME="/mnt/public/gaojiahuan/huggingface/cache/${HOSTNAME}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

export PATH=/usr/local/cuda/bin:$PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

export CONFIG_NAME="bagging_sku_0824_sm2sm"

# 归一化
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}

export WANDB_PROJECT="openpi"
export WANDB_ENTITY="striding-algo"

# 训练（还可以设定 WANDB_API_KEY参数）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python3 scripts/train.py ${CONFIG_NAME} \
--overwrite --data.random-drop-master 0.10 --data.random-drop-history 0.50 \
--data.random-drop-future 0.50
