"""一次性脚本：扫描候选设计，找"骑在点火悬崖边"的那一发（P(点火) 落在 40~80%）。"""
import sys

import numpy as np

sys.path.insert(0, ".")
from cogsim.simulator import PERTURB_SCALE_PRIOR, logYn_to_MJ, simulate


def sample(x, mode, n=1500, seed=0):
    v = []
    for m in range(n):
        d = np.random.default_rng(seed + m).normal(0, 1, 4) * PERTURB_SCALE_PRIOR
        s, _ = simulate(x, d, np.random.default_rng(seed + m), mode=mode)
        v.append(logYn_to_MJ(s[0]))
    return np.asarray(v)


cands = {
    "NOMINAL": np.array([2.05, 440., 1050., 80., 80., 0.30, 0.60, 0.0, 80.]),
    "NEW_a": np.array([2.15, 470., 1080., 80., 86., 0.25, 0.55, -0.5, 60.]),
    "NEW_b": np.array([2.15, 460., 1060., 82., 84., 0.28, 0.60, 0.0, 70.]),
    "NEW_c": np.array([2.10, 450., 1050., 80., 84., 0.30, 0.60, 0.0, 75.]),
    "NEW_d": np.array([2.15, 455., 1055., 81., 85., 0.30, 0.60, 0.0, 65.]),
    "NEW_e": np.array([2.15, 470., 1075., 78., 88., 0.22, 0.55, 0.0, 55.]),
    "NEW_f": np.array([2.18, 470., 1075., 79., 88., 0.22, 0.55, 0.0, 55.]),
}
prev = np.array([1.90, 430., 1040., 78., 78., 0.35, 0.70, 0.50, 110.])
cands["PREV"] = prev

for k, x in cands.items():
    v = sample(x, "exp")
    p = float((v > x[0]).mean())
    print(f"{k:8s} E={x[0]:.2f}  中位={np.median(v):6.3f} MJ  "
          f"90%CI=[{np.quantile(v,0.05):.2f},{np.quantile(v,0.95):.2f}]  "
          f"P(点火|真值)={p*100:5.1f}%")
