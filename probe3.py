"""Compare the two candidate fixes for the 0824 blow-up."""
import dataclasses
import numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _dl
from openpi import transforms as _transforms
from openpi.shared import normalize as _normalize

BASE = _config.get_config("make_whiskey_sour_sm2sm")
REPO = "make_whiskey_sour_ccw_0824_sft_move_jitter"
N = 300


def probe(floor=None, **over):
    data = dataclasses.replace(BASE.data, repo_id=REPO, **over)
    cfg = dataclasses.replace(BASE, data=data)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    if floor is not None and dc.norm_stats is not None:
        ns = dict(dc.norm_stats)
        for k in ("state", "actions"):
            st = ns[k]
            ns[k] = _normalize.NormStats(
                mean=st.mean, std=np.maximum(np.asarray(st.std), floor),
                q01=st.q01, q99=st.q99)
        dc = dataclasses.replace(dc, norm_stats=ns)
    ds = _dl.transform_dataset(
        _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model), dc)
    idx = np.linspace(0, len(ds) - 1, min(N, len(ds))).astype(int)
    S = np.stack([np.asarray(ds[int(i)]["state"]) for i in idx])
    A = np.stack([np.asarray(ds[int(i)]["actions"]) for i in idx])
    return S, A


for tag, kw in [("current (offset 0.020)", {}),
                ("fix A: random_pos_offset = 0", {"random_pos_offset": 0.0}),
                ("fix B: std floor 0.01, keep offset", {"floor": 0.01}),
                ("fix B': std floor 0.005, keep offset", {"floor": 0.005})]:
    S, A = probe(**kw)
    pd_ = (S ** 2).reshape(-1, S.shape[-1]).mean(axis=0)
    print(f"  {tag:<38} state[{S.min():+8.2f},{S.max():+8.2f}]  "
          f"loss~{1 + (A**2).mean():7.2f}   worst dim {pd_.argmax()}")
