"""Run openpi's own pipeline and measure what actually reaches the model."""
import dataclasses, sys
import numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _dl

BASE = _config.get_config("make_whiskey_sour_sm2sm")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

for repo in ["make_whiskey_sour_ccw_0824_sft_move_jitter",
             "make_whiskey_sour_ccw_0817_sft_move_jitter",
             "make_whiskey_sour_ccw_0820_sft_move_jitter"]:
    data = dataclasses.replace(BASE.data, repo_id=repo)
    cfg = dataclasses.replace(BASE, data=data)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = _dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    ds = _dl.transform_dataset(ds, dc)
    idx = np.linspace(0, len(ds) - 1, min(N, len(ds))).astype(int)
    S, A = [], []
    for i in idx:
        s = ds[int(i)]
        S.append(np.asarray(s["state"]))
        A.append(np.asarray(s["actions"]))
    S = np.stack(S).reshape(len(idx), -1)
    A = np.stack(A).reshape(len(idx), -1)
    print(f"\n{'='*80}\n{repo}   n={len(idx)}")
    print(f"  state   shape={np.asarray(ds[0]['state']).shape}  "
          f"min={S.min():+.2f} max={S.max():+.2f}  "
          f"mean|s|={np.abs(S).mean():.3f}  p99.9|s|={np.percentile(np.abs(S),99.9):.2f}")
    print(f"  actions shape={np.asarray(ds[0]['actions']).shape}  "
          f"min={A.min():+.2f} max={A.max():+.2f}  "
          f"mean|a|={np.abs(A).mean():.3f}  p99.9|a|={np.percentile(np.abs(A),99.9):.2f}")
    print(f"  implied flow loss  E[(noise-a)^2] ~ 1 + mean(a^2) = {1 + (A**2).mean():.2f}")
    a2 = (A ** 2).reshape(len(idx), -1)
    worst = np.argsort(-a2.mean(axis=0))[:6]
    print(f"  worst action slots (flat idx): " +
          " ".join(f"{w}:{a2[:,w].mean():.1f}" for w in worst))
    s2 = (S ** 2)
    ws = np.argsort(-s2.mean(axis=0))[:6]
    print(f"  worst state slots  (flat idx): " +
          " ".join(f"{w}:{s2[:,w].mean():.1f}" for w in ws))
