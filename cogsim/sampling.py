"""设计空间采样：拉丁超立方 (LHS)。

对应 LLNL 侧"用 Merlin 在 Sierra 上撒 10^5~10^8 个仿真点"的第一步。
这里规模缩到 10^3 量级，但采样原则一致：在给定设计空间上做空间填充采样。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .simulator import DESIGN_BOUNDS, N_DESIGN


def lhs_designs(n: int, bounds: np.ndarray = DESIGN_BOUNDS, seed: int = 0) -> np.ndarray:
    """在 [lo, hi] 超立方体内生成 n 个 LHS 设计点，返回 (n, dim)。"""
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed, optimization="random-cd")
    u = sampler.random(n=n)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def sample_perturbations(n: int, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按工程先验采样打靶涨落 δ ~ N(0, diag(scale))。"""
    return rng.normal(0.0, 1.0, size=(n, scale.size)) * scale[None, :]


def normalize_designs(x: np.ndarray, bounds: np.ndarray = DESIGN_BOUNDS) -> np.ndarray:
    return 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0


def denormalize_designs(u: np.ndarray, bounds: np.ndarray = DESIGN_BOUNDS) -> np.ndarray:
    return 0.5 * (u + 1.0) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
