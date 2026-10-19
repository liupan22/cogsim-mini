"""逆向设计与稳健性优化。

CogSim 不只做"预测"，还反过来回答：**应该打哪一发？**

关键结论（Science 2025 论文的要点之一）：
    最大化**平均产额** ≠ 最大化**点火概率**。
平均产额最高的设计往往骑在点火悬崖边缘，打靶涨落一来就掉下去；
真正该优化的是 P(Y > 阈值)，即**稳健性指标**（robustness metric）。

本模块给出两种目标的对比：
    naive  : argmax E[Y|δ=0]
    robust : argmax P(Y > Y_threshold | 全部不确定度)
"""
from __future__ import annotations

import numpy as np

from .bayes import IDX_LOGY, jacobian_logy
from .models.transfer import ensemble_predict_scalars
from .simulator import (  # noqa: E402
    DESIGN_BOUNDS, PERTURB_SCALE_PRIOR, SCALAR_NAMES, MJ_to_logYn,
)


def _ens_logy(models, designs, scaler_in, scaler_obs):
    """集成预测 log10 产额：返回 (均值, 成员间标准差=认知不确定度)。"""
    d = np.atleast_2d(np.asarray(designs, dtype=np.float64))
    Z = np.hstack([d, np.zeros((len(d), len(PERTURB_SCALE_PRIOR)))])
    out = ensemble_predict_scalars(models, scaler_in(Z), scaler_obs)
    return out["mean"][:, IDX_LOGY], out["std"][:, IDX_LOGY]


def optimize(models, posterior, scaler_in, scaler_obs,
             threshold_mj: float,
             bounds: np.ndarray = DESIGN_BOUNDS,
             fixed: dict | None = None,
             n_screen: int = 6000,
             n_shortlist: int = 200,
             n_final: int = 20,
             sigma_meas: float = 0.035,
             trust_ref_designs: np.ndarray | None = None,
             trust_quantile: float = 0.90,
             margin: float = 0.0,
             center: np.ndarray | None = None,
             radius: float = 1.0,
             seed: int = 0) -> dict:
    """三级漏斗式搜索：粗筛(平均产额) → 线性化稳健性排序 → 蒙特卡洛精算。

    fixed: {维度下标: 固定值}，用于施加硬约束（例如装置激光能量上限 E_laser ≤ 2.0 MJ）。
           真实设计任务里，能量/功率这类"设施约束"通常是先钉死的。

    可信域（trust region）：候选设计的集成离散度不得超过"held-out 仿真集上的
    典型离散度"。没有这一步，优化器会跑到训练数据稀疏的角落，找到代理模型
    "吹出来"的假最优解 —— 这是代理优化最经典的翻车方式。
    """
    from .bayes import propagate
    from scipy.special import ndtr

    fixed = fixed or {}
    rng = np.random.default_rng(seed)
    span = bounds[:, 1] - bounds[:, 0]
    # margin：不贴边界搜索（真实设计空间总要留工程余量，而且边界正是
    # 训练样本最稀疏、代理最容易外推的地方）
    lo, hi = bounds[:, 0].copy() + margin * span, bounds[:, 1].copy() - margin * span
    # center/radius：设计空间的局部可信域。真实型号演进是在当前基线附近迭代，
    # 不会跳到任意角落；放开全空间搜索，优化器必然钻进代理误差为正的角落。
    if center is not None:
        c = np.asarray(center, dtype=np.float64)
        lo = np.maximum(lo, c - radius * span)
        hi = np.minimum(hi, c + radius * span)
    for k, v in fixed.items():
        lo[k] = hi[k] = float(v)
    U = rng.uniform(0, 1, size=(n_screen, bounds.shape[0]))
    cand = lo + U * (hi - lo)

    # ---- A. 粗筛 + 可信域过滤（参照 held-out 仿真集上代理的离散度水平）
    f0_all, epi_all = _ens_logy(models, cand, scaler_in, scaler_obs)
    if trust_ref_designs is not None:
        _, epi_ref = _ens_logy(models, trust_ref_designs, scaler_in, scaler_obs)
    else:
        epi_ref = epi_all
    epi_max = float(np.quantile(epi_ref, trust_quantile))
    trust = np.flatnonzero(epi_all <= epi_max)
    if trust.size < 50:            # 兜底：过滤过狠时保留最好的一批
        trust = np.argsort(epi_all)[:50]
    cand, f0 = cand[trust], f0_all[trust]

    naive_idx = int(np.argmax(f0))
    order = np.argsort(-f0)[:n_shortlist]
    short = cand[order]

    # ---- B. 线性化稳健性打分：p ≈ Φ((f+μ−log10 thr) / sd)
    J = jacobian_logy(models, short, scaler_in, scaler_obs)
    s_med = float(np.median(posterior["s"]))
    mu_med = float(np.median(posterior["mu"]))
    sd = np.sqrt(s_med ** 2 * ((J * PERTURB_SCALE_PRIOR[None, :]) ** 2).sum(axis=1) + sigma_meas ** 2)
    thr_logyn = float(MJ_to_logYn(threshold_mj))
    p_lin = ndtr((f0[order] + mu_med - thr_logyn) / (sd + 1e-9))
    final_idx = order[np.argsort(-p_lin)[:n_final]]
    finals = cand[final_idx]

    # ---- C. 蒙特卡洛精算
    res_n = propagate(models, cand[naive_idx:naive_idx + 1], posterior, scaler_in, scaler_obs,
                      threshold_mj, n_mc=256, n_post=80, seed=seed + 1)
    res_f = propagate(models, finals, posterior, scaler_in, scaler_obs,
                      threshold_mj, n_mc=256, n_post=80, seed=seed + 2)
    k_best = int(np.argmax(res_f["p_ignition"]))

    naive_design = cand[naive_idx]
    robust_design = finals[k_best]
    return {
        "naive_design": naive_design,
        "robust_design": robust_design,
        "naive": {"p_ignition": float(res_n["p_ignition"][0]),
                  "mean_logy": float(f0[naive_idx]),
                  "yield_p05": float(res_n["yield_p05_MJ"][0]),
                  "yield_p50": float(res_n["yield_p50_MJ"][0]),
                  "yield_p95": float(res_n["yield_p95_MJ"][0])},
        "robust": {"p_ignition": float(res_f["p_ignition"][k_best]),
                   "mean_logy": float(_ens_logy(models, robust_design, scaler_in, scaler_obs)[0][0]),
                   "yield_p05": float(res_f["yield_p05_MJ"][k_best]),
                   "yield_p50": float(res_f["yield_p50_MJ"][k_best]),
                   "yield_p95": float(res_f["yield_p95_MJ"][k_best])},
        "trust_epi_max": epi_max,
        "n_trusted": int(len(trust)),
        "candidates": cand,
        "screen_logy": f0,
        "shortlist_p_lin": p_lin,
    }
