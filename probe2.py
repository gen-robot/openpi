"""Decisive test: is random_pos_offset=0.020 the whole story for 0824?"""
import dataclasses, sys
import numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _dl

BASE = _config.get_config("make_whiskey_sour_sm2sm")
N = 300


def probe(repo, **over):
    data = dataclasses.replace(BASE.data, repo_id=repo, **over)
    cfg = dataclasses.replace(BASE, data=data)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = _dl.transform_dataset(
        _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model), dc)
    idx = np.linspace(0, len(ds) - 1, min(N, len(ds))).astype(int)
    S = np.stack([np.asarray(ds[int(i)]["state"]) for i in idx])
    A = np.stack([np.asarray(ds[int(i)]["actions"]) for i in idx])
    return S, A


def rep(tag, S, A):
    per_dim = (S ** 2).reshape(-1, S.shape[-1]).mean(axis=0)
    print(f"  {tag:<34} state[{S.min():+9.2f},{S.max():+9.2f}]  "
          f"loss~{1 + (A**2).mean():8.2f}  worst dim={per_dim.argmax()} "
          f"(E[s^2]={per_dim.max():.1f})")


print("0824  (right arm frozen, slave std ~1e-4)")
rep("as configured (offset 0.020)", *probe("make_whiskey_sour_ccw_0824_sft_move_jitter"))
rep("random_pos_offset = 0", *probe("make_whiskey_sour_ccw_0824_sft_move_jitter",
                                    random_pos_offset=0.0))
print("\n0817  (left arm frozen, slave std ~5e-3)")
rep("as configured (offset 0.020)", *probe("make_whiskey_sour_ccw_0817_sft_move_jitter"))
rep("random_pos_offset = 0", *probe("make_whiskey_sour_ccw_0817_sft_move_jitter",
                                    random_pos_offset=0.0))
