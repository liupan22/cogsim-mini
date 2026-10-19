"""评估指标：与 LLNL 论文口径一致的"平均相对误差"与 R²。"""
from __future__ import annotations

import numpy as np

from .simulator import SCALAR_NAMES


def relative_error(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """逐列平均相对误差 mean(|pred-obs| / |obs|)。

    对 log10_Y 这种"本身就是对数"的量，LLNL 用 |Δ|/|obs| 直接算，
    这里保持一致；对可能接近 0 的量加一个分母下限防止爆炸。
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    denom = np.maximum(np.abs(obs), 1e-3)
    return np.mean(np.abs(pred - obs) / denom, axis=0)


def dex_error(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """对数类观测量（log10 产额）的误差：|Δ| 直接就是"差多少个数量级"(dex)。"""
    return np.mean(np.abs(np.asarray(pred) - np.asarray(obs)), axis=0)


def mixed_error(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """对数量纲用 dex，其余用相对误差。"""
    rel = relative_error(pred, obs)
    dex = dex_error(pred, obs)
    return np.where(np.array([n.startswith("log10") for n in SCALAR_NAMES]), dex, rel)


def r2_score(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    ss_res = ((pred - obs) ** 2).sum(axis=0)
    ss_tot = ((obs - obs.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def table(err_sim: np.ndarray, err_tl: np.ndarray) -> str:
    """生成对比表（文本）。log10 产额这一行用 dex 口径，其余用相对误差。"""
    head = f"{'observable':<14}{'sim-only':>13}{'sim+transfer':>15}{'改善':>10}"
    lines = [head, "-" * len(head)]
    for i, name in enumerate(SCALAR_NAMES):
        a, b = err_sim[i], err_tl[i]
        if name.startswith("log10"):
            lines.append(f"{name+'[dex]':<14}{a:>13.3f}{b:>15.3f}{(a-b):>10.3f}")
        else:
            a, b = a * 100, b * 100
            lines.append(f"{name+'[%]':<14}{a:>13.2f}{b:>15.2f}{(a-b):>10.2f}")
    lines.append("-" * len(head))
    m1 = err_sim[[i for i, n in enumerate(SCALAR_NAMES) if not n.startswith("log10")]].mean() * 100
    m2 = err_tl[[i for i, n in enumerate(SCALAR_NAMES) if not n.startswith("log10")]].mean() * 100
    lines.append(f"{'MEAN(non-log)[%]':<14}{m1:>13.2f}{m2:>15.2f}{(m1-m2):>10.2f}")
    return "\n".join(lines)
