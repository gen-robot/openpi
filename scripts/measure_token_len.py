"""Measure the true max token length for a training config's prompt+state input."""
import collections
import logging

import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

CONFIG_NAME = "bagging_sku_0811_pi05_sm2sm"
N_SAMPLES = 20000

logging.basicConfig(level=logging.ERROR)

cfg = _config.get_config(CONFIG_NAME)
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model, split=None)
print(f"dataset size: {len(dataset)}", flush=True)

skip_types = (_transforms.TokenizePrompt, _transforms.BuildStateSequence, _transforms.PadStatesAndActions)
model_inputs = [t for t in data_config.model_transforms.inputs if not isinstance(t, skip_types)]
transforms = [
    *data_config.repack_transforms.inputs,
    *data_config.data_transforms.inputs,
    _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
    *model_inputs,
]
tds = _data_loader.TransformedDataset(dataset, transforms)

tok = _tokenizer.PaligemmaTokenizer(max_len=100000)

loader = torch.utils.data.DataLoader(
    tds, batch_size=None, shuffle=True, num_workers=16, prefetch_factor=4, persistent_workers=False
)

lens = []
prompt_lens = collections.Counter()
max_len = 0
max_prompt = None
state_shape_seen = set()

for i, item in enumerate(loader):
    if i >= N_SAMPLES:
        break
    prompt = item.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        prompt = prompt.item()
    state = item.get("state")
    if state is not None:
        state = np.asarray(state)
        state_shape_seen.add(tuple(state.shape))
    tokens, mask = tok.tokenize(prompt, state)
    n = int(np.sum(mask))
    lens.append(n)
    p_tokens, p_mask = tok.tokenize(prompt, None)
    prompt_lens[int(np.sum(p_mask))] += 1
    if n > max_len:
        max_len = n
        max_prompt = prompt
    if (i + 1) % 1000 == 0:
        arr = np.asarray(lens)
        print(f"[{i+1}] max={arr.max()} p99={np.percentile(arr, 99):.0f} mean={arr.mean():.0f}", flush=True)

arr = np.asarray(lens)
print("=" * 60, flush=True)
print(f"samples measured : {len(arr)}")
print(f"state shapes     : {state_shape_seen}")
print(f"max token len    : {arr.max()}")
print(f"p99.9            : {np.percentile(arr, 99.9):.0f}")
print(f"p99              : {np.percentile(arr, 99):.0f}")
print(f"mean             : {arr.mean():.0f}")
print(f"max prompt-only  : {max(prompt_lens) if prompt_lens else None}")
print(f"longest prompt   : {max_prompt!r}")
