"""贝叶斯标定 + 不确定度传播 —— CogSim 真正"下判断"的那一步。

对应文献
--------
Spears et al. (LLNL), "Predicting fusion ignition at the National Ignition
Facility with physics-informed deep learning", Science 389, 727 (2025).

这套方法在 2022-12-05 的 N221204 打靶前给出"点火概率 > 70%"，
事后实测产额落在预测区间内。

要回答的问题不是"这一发能出多少产额"，而是：
    **在已知的所有不确定度下，这一发点火的概率是多少？**

不确定度分三类
--------------
1. **打靶涨落（aleatoric）**  激光能量精度、靶丸缺陷、驱动不对称、充气密度
   每发都不一样 —— 用一个涨落向量 δ 的分布描述；
2. **认知不确定度（epistemic）** 代理模型本身没学准 —— 用深度集成的成员离散度描述；
3. **残余系统偏差**  仿真模型即使在迁移学习后仍有偏差 —— 用一个待推断的偏移 μ 描述。

流程
----
  (a) 用代理模型对每发历史实验求产额对 δ 的敏感度 Jacobian；
  (b) 以历史实验的残差为似然，MCMC 推断 (涨落幅度 s, 系统偏移 μ) 的后验；
  (c) 对新设计做蒙特卡洛传播：抽 δ ~ N(0,(s·w)²) → 过代理 → 得到产额分布
      → P(点火) = P(Y > Y_threshold)。
"""
from __future__ import annotations

import numpy as np
import torch

from .simulator import (  # noqa: E402
    N_DESIGN, PERTURB_SCALE_PRIOR, SCALAR_NAMES, MJ_to_logYn, logYn_to_MJ,
)

IDX_LOGY = SCALAR_NAMES.index("log10_Yn")


# ---------------------------------------------------------------- 敏感度
@torch.no_grad()
def jacobian_logy(models, designs: np.ndarray, scaler_in, scaler_obs,
                  eps: np.ndarray | None = None) -> np.ndarray:
    """∂(log10 Y) / ∂δ，在 δ = 0 处中心差分。返回 (n, n_perturb)。"""
    from .models.transfer import ensemble_predict_scalars

    d = np.asarray(designs, dtype=np.float64)
    n_p = len(PERTURB_SCALE_PRIOR)
    if eps is None:
        eps = np.maximum(PERTURB_SCALE_PRIOR * 0.25, 1e-4)

    J = np.zeros((len(d), n_p))
    for i in range(n_p):
        step = np.zeros(n_p)
        step[i] = eps[i]
        zp = np.hstack([d, np.tile(step, (len(d), 1))])
        zm = np.hstack([d, np.tile(-step, (len(d), 1))])
        yp = ensemble_predict_scalars(models, scaler_in(zp), scaler_obs)["mean"][:, IDX_LOGY]
        ym = ensemble_predict_scalars(models, scaler_in(zm), scaler_obs)["mean"][:, IDX_LOGY]
        J[:, i] = (yp - ym) / (2 * eps[i])
    return J


# ---------------------------------------------------------------- MCMC 标定
def calibrate_variability(J: np.ndarray,
                          residual: np.ndarray,
                          prior_scale: np.ndarray = PERTURB_SCALE_PRIOR,
                          sigma_meas: float = 0.035,
                          n_iter: int = 6000,
                          burn: int = 2000,
                          thin: int = 4,
                          seed: int = 0) -> dict:
    """推断 (log s, μ)。

    s  : 打靶涨落幅度的全局缩放（相对工程先验）
    μ  : 迁移学习后残余的系统偏移（log10 产额）

    似然： r_j ~ N(μ, s²·Σ_i(J_ji·w_i)² + σ_meas²)
    """
    rng = np.random.default_rng(seed)
    base = (J * prior_scale[None, :]) ** 2          # (n, n_p)  s=1 时的方差贡献
    base_var = base.sum(axis=1)                     # (n,)

    def log_post(theta):
        ls, mu = theta
        s2 = np.exp(2.0 * ls)
        var = s2 * base_var + sigma_meas ** 2
        ll = -0.5 * np.sum((residual - mu) ** 2 / var + np.log(var))
        lp = -0.5 * (ls / 0.7) ** 2 - 0.5 * (mu / 0.5) ** 2
        return ll + lp

    theta = np.array([0.0, float(np.mean(residual))])
    step = np.array([0.25, 0.08])
    cur = log_post(theta)
    chain, acc = [], 0
    for it in range(n_iter):
        prop = theta + step * rng.normal(size=2)
        lp = log_post(prop)
        if np.log(rng.uniform()) < lp - cur:
            theta, cur = prop, lp
            acc += 1
        if it >= burn and (it - burn) % thin == 0:
            chain.append(theta.copy())
    chain = np.asarray(chain)
    return {
        "s": np.exp(chain[:, 0]),
        "mu": chain[:, 1],
        "accept_rate": acc / n_iter,
        "chain": chain,
    }


# ---------------------------------------------------------------- 传播
@torch.no_grad()
def propagate(models, designs: np.ndarray, posterior: dict, scaler_in, scaler_obs,
              y_threshold_mj: float | np.ndarray,
              prior_scale: np.ndarray = PERTURB_SCALE_PRIOR,
              sigma_meas: float = 0.035,
              n_mc: int = 256,
              n_post: int = 120,
              seed: int = 0,
              epistemic: float | None = None) -> dict:
    """把不确定度传播过代理模型，得到产额分布与点火概率。

    返回每个设计的：点火概率、产额分位数、以及涨落贡献/认知贡献的分解。
    """
    from .models.transfer import ensemble_predict_scalars

    designs = np.atleast_2d(np.asarray(designs, dtype=np.float64))
    rng = np.random.default_rng(seed)
    s_all, mu_all = posterior["s"], posterior["mu"]
    idx = rng.choice(len(s_all), size=min(n_post, len(s_all)), replace=False)
    thr = np.atleast_1d(np.asarray(y_threshold_mj, dtype=np.float64))

    Y = []          # 每次 MC 的 log10 产额样本
    for k in idx:
        s, mu = s_all[k], mu_all[k]
        delta = rng.normal(0.0, 1.0, size=(n_mc, len(prior_scale))) * (s * prior_scale)[None, :]
        X = np.hstack([np.repeat(designs, n_mc, axis=0),
                       np.tile(delta, (len(designs), 1))])
        pred = ensemble_predict_scalars(models, scaler_in(X), scaler_obs)
        y = pred["mean"][:, IDX_LOGY] + mu
        e_std = pred["std"][:, IDX_LOGY] if epistemic is None else epistemic
        y = y + e_std * rng.normal(size=y.shape)
        y = y + sigma_meas * rng.normal(size=y.shape)
        Y.append(y.reshape(len(designs), n_mc))

    Y = np.concatenate(Y, axis=1)                       # (n_design, n_mc*n_post)  [log10 Yn]
    thr_c = thr if thr.size == len(designs) else np.repeat(thr, len(designs))
    thr_logyn = np.atleast_1d(MJ_to_logYn(thr_c))       # 阈值换成 log10(中子数)
    p_ign = (Y > thr_logyn[:, None]).mean(axis=1)
    mj = logYn_to_MJ(Y)
    return {
        "p_ignition": p_ign,
        "samples": Y,
        "samples_MJ": mj,
        "yield_p05_MJ": np.quantile(mj, 0.05, axis=1),
        "yield_p50_MJ": np.quantile(mj, 0.50, axis=1),
        "yield_p95_MJ": np.quantile(mj, 0.95, axis=1),
        "logy_std": Y.std(axis=1),
        "threshold_MJ": thr_c,
    }
