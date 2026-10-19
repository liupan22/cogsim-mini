"""mini-CogSim 全流程：从"跑仿真"到"给出点火概率"和"推荐下一发"。

运行：
    python run_pipeline.py                # 默认规模（约 3-6 分钟）
    python run_pipeline.py --quick        # 快速冒烟（约 1 分钟）
    python run_pipeline.py --n-sim 6000   # 更大规模

流水线（每个 stage 都对应 LLNL CogSim 的一个环节）：
    S1  大规模仿真数据库          （Merlin + HYDRA）
    S2  历史实验数据              （NIF 打靶记录）
    S3  MaCC 多模态代理训练        （流形一致 + 循环一致）
    S4  迁移学习 + 深度集成        （仿真 → 实验校正）
    S5  贝叶斯标定 + 不确定度传播   （点火概率）
    S6  稳健性逆向设计             （下一发打什么）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cogsim import bayes, design as dsign, metrics, report  # noqa: E402
from cogsim.dataset import InputScaler, ObsScaler, ObsView, to_tensor  # noqa: E402
from cogsim.models.macc import MaCC, predict_scalars, train_macc  # noqa: E402
from cogsim.models.transfer import deep_ensemble_finetune, ensemble_predict_scalars  # noqa: E402
from cogsim.sampling import lhs_designs, sample_perturbations  # noqa: E402
from cogsim.simulator import (  # noqa: E402
    DESIGN_BOUNDS, N_DESIGN, PERTURB_SCALE_PRIOR, SCALAR_NAMES, MEAS_SIGMA,
    simulate, logYn_to_MJ, MJ_to_logYn,
)
from cogsim import workflow  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# ------------------------------------------------------------------ 配置
CFG = dict(
    n_sim=12000, n_sim_test=800,
    n_exp=40, n_exp_train=25,
    epochs_ae=120, epochs_joint=200,
    tl_epochs=400, n_members=4,
    # 任务门槛：产额 > 激光能量（NAS 点火判据，增益 > 1）
    mission_threshold_mj=2.15,
    energy_budget_mj=2.15,        # 设施硬约束：激光能量上限 2.15 MJ（设计优化时钉死）
    seed=7,
)

# 两个待评估的设计：[E_laser, P_peak, R_out, dR_abl, dR_ice, dopant, gas, p2, rough]
PREV_DESIGN = np.array([1.90, 430.0, 1040.0, 78.0, 78.0, 0.35, 0.70, 0.50, 110.0])
# 新一代设计：用 tools/scan_design.py 扫描挑出的"骑在点火悬崖边"的一发
# （真值 P(点火)≈52%，中位产额≈2.17 MJ，阈值 2.15 MJ —— 与 N221204 的局面类似）
NEW_DESIGN = np.array([2.15, 470.0, 1075.0, 78.0, 88.0, 0.22, 0.55, 0.00, 55.0])


def log(msg):
    print(msg, flush=True)


def true_p_ignition(design, threshold_mj, n=800, seed=0):
    """用真实"求解器"蒙特卡洛出的点火概率 —— 只在合成实验里才拿得到，
    用来检验概率预测的标定好坏。"""
    ys = []
    for m in range(n):
        dlt = sample_perturbations(1, PERTURB_SCALE_PRIOR, np.random.default_rng(seed + m))[0]
        s, _ = simulate(design, dlt, np.random.default_rng(seed + m), mode="exp")
        ys.append(s[bayes.IDX_LOGY])
    mj = logYn_to_MJ(np.asarray(ys))
    return float((mj > threshold_mj).mean()), float(np.median(mj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=CFG["n_sim"])
    ap.add_argument("--epochs-ae", type=int, default=CFG["epochs_ae"])
    ap.add_argument("--epochs-joint", type=int, default=CFG["epochs_joint"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_sim, a.epochs_ae, a.epochs_joint = 900, 40, 60
        CFG["tl_epochs"], CFG["n_members"] = 150, 3

    t_all = time.time()
    rng = np.random.default_rng(CFG["seed"])
    os.makedirs(OUT, exist_ok=True)
    figs_dir = os.path.join(OUT, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    log("=" * 78)
    log(" mini-CogSim  |  认知仿真流水线")
    log("=" * 78)

    # ------------------------------------------------------------ S1+S3b 仿真数据库（含自适应补样）
    log("\n[S1] 生成大规模仿真数据库（对应 Merlin + HYDRA）")
    n_sim = a.n_sim
    n_ada = int(0.38 * n_sim)
    n_lhs = n_sim - n_ada
    sim_designs = lhs_designs(n_lhs, seed=1)
    sim_perturbs = sample_perturbations(n_lhs, PERTURB_SCALE_PRIOR, rng)
    t0 = time.time()
    workflow.run_ensemble(sim_designs, sim_perturbs, OUT, "simdb_lhs", mode="sim", seed0=1000)
    sim_t_per_run = (time.time() - t0) / n_lhs
    z = np.load(os.path.join(OUT, "simdb_lhs.npz"))
    X0, S0, I0 = z["inputs"], z["scalars"], z["images"]
    log(f"     LHS 阶段 {n_lhs} 次仿真")

    # ---- S3b 自适应补样：点火悬崖附近才是决策关键区，均匀采样在那里样本太少。
    #      先用一轮粗代理定位悬崖位置，再把样本补进去（CogSim 的主动学习闭环）。
    log("\n[S3b] 自适应补样：在点火悬崖附近加密采样")
    sc0 = ObsScaler().fit(S0, I0)
    in0 = InputScaler().fit(X0)
    model0 = MaCC(X0.shape[1], sc0.obs_dim, z_dim=32, hidden=512, view=ObsView())
    train_macc(model0, to_tensor(in0(X0)), to_tensor(sc0.encode(S0, I0)),
               epochs_ae=60, epochs_joint=100, lr=1e-3, batch_size=256,
               seed=CFG["seed"], verbose=False)
    thr_logyn = float(MJ_to_logYn(CFG["mission_threshold_mj"]))
    U = DESIGN_BOUNDS[:, 0] + rng.uniform(0, 1, size=(20000, N_DESIGN)) * (
        DESIGN_BOUNDS[:, 1] - DESIGN_BOUNDS[:, 0])
    p0 = predict_scalars(model0, to_tensor(in0(np.hstack([U, np.zeros((len(U), 4))]))),
                         sc0)[:, bayes.IDX_LOGY]
    near = np.flatnonzero(np.abs(p0 - thr_logyn) < 0.30)
    if near.size < n_ada:
        near = np.argsort(np.abs(p0 - thr_logyn))[:n_ada]
    pick = rng.choice(near, size=min(n_ada, near.size), replace=False)
    ada_designs = U[pick]
    ada_perturbs = sample_perturbations(len(ada_designs), PERTURB_SCALE_PRIOR, rng)
    workflow.run_ensemble(ada_designs, ada_perturbs, OUT, "simdb_ada", mode="sim", seed0=300000)
    z2 = np.load(os.path.join(OUT, "simdb_ada.npz"))
    log(f"     在 |log10Y - 阈值| < 0.45 dex 的悬崖带内补 {len(ada_designs)} 次仿真")

    X_sim = np.vstack([X0, z2["inputs"]])
    S_sim = np.vstack([S0, z2["scalars"]])
    I_sim = np.vstack([I0, z2["images"]])
    log(f"     仿真库合计 {len(X_sim)} 次，落盘 outputs/simdb_*.npz")

    idx = rng.permutation(n_sim)
    te, tr = idx[: CFG["n_sim_test"]], idx[CFG["n_sim_test"]:]
    Xtr, Str, Itr = X_sim[tr], S_sim[tr], I_sim[tr]
    Xte, Ste, Ite = X_sim[te], S_sim[te], I_sim[te]

    # ------------------------------------------------------------ S2 历史实验
    log("\n[S2] 生成历史实验数据（对应 NIF 打靶记录）")
    n_exp = CFG["n_exp"]
    center, half = PREV_DESIGN, 0.35 * (DESIGN_BOUNDS[:, 1] - DESIGN_BOUNDS[:, 0])
    lo = np.maximum(center - half, DESIGN_BOUNDS[:, 0])
    hi = np.minimum(center + half, DESIGN_BOUNDS[:, 1])
    u = rng.uniform(0, 1, size=(n_exp, N_DESIGN))
    exp_designs = lo + u * (hi - lo)
    exp_perturbs_true = sample_perturbations(n_exp, PERTURB_SCALE_PRIOR, rng)  # 真实涨落
    exp_S, exp_I = [], []
    for i in range(n_exp):
        s, im = simulate(exp_designs[i], exp_perturbs_true[i],
                         np.random.default_rng(5000 + i), mode="exp")
        exp_S.append(s)
        exp_I.append(im)
    exp_S = np.stack(exp_S)
    log(f"     历史实验 {n_exp} 发：{CFG['n_exp_train']} 发用于迁移学习，"
        f"{n_exp - CFG['n_exp_train']} 发留作盲测")

    # ------------------------------------------------------------ 标准化
    scaler_obs = ObsScaler().fit(Str, Itr)
    scaler_in = InputScaler().fit(Xtr)
    Obs_tr = to_tensor(scaler_obs.encode(Str, Itr))
    Obs_te = to_tensor(scaler_obs.encode(Ste, Ite))
    Xtr_t, Xte_t = to_tensor(scaler_in(Xtr)), to_tensor(scaler_in(Xte))

    # ------------------------------------------------------------ S3 代理训练
    log("\n[S3] 训练 MaCC 多模态代理（自编码器 + 前向 + 逆向）")
    obs_dim = scaler_obs.obs_dim
    torch.manual_seed(CFG["seed"])
    model = MaCC(X_sim.shape[1], obs_dim, z_dim=32, hidden=512, view=ObsView())
    t0 = time.time()
    train_macc(model, Xtr_t, Obs_tr, epochs_ae=a.epochs_ae, epochs_joint=a.epochs_joint,
               lr=1e-3, batch_size=256, seed=CFG["seed"])
    log(f"     训练用时 {time.time()-t0:.1f}s")

    # 代理 vs 仿真：精度与加速比
    pred_te = predict_scalars(model, Xte_t, scaler_obs)
    err_surrogate = metrics.mixed_error(pred_te, Ste)
    r2_surrogate = metrics.r2_score(pred_te, Ste)
    t0 = time.time()
    _ = predict_scalars(model, Xte_t[:1000], scaler_obs)
    sur_t_per_run = (time.time() - t0) / min(1000, len(Xte_t))
    speedup = sim_t_per_run / max(sur_t_per_run, 1e-9)
    log("     代理 vs 仿真（held-out 仿真集）:")
    for n_, e_, r_ in zip(SCALAR_NAMES, err_surrogate, r2_surrogate):
        unit = "dex" if n_.startswith("log10") else "%"
        log(f"       {n_:<14} err={e_*100 if unit=='%' else e_:6.3f}{unit:>4}   R2={r_:6.3f}")
    log(f"     加速比 ≈ {speedup:.0f}x  "
        f"(仿真 {sim_t_per_run*1e3:.2f} ms/发 vs 代理 {sur_t_per_run*1e6:.2f} µs/发)")

    # ------------------------------------------------------------ S4 迁移学习
    log("\n[S4] 迁移学习：用少量实验校正仿真代理（对应 Humbird et al. 2021）")
    Xexp_all = np.hstack([exp_designs, np.zeros((n_exp, len(PERTURB_SCALE_PRIOR)))])
    S_exp_std = to_tensor((exp_S - scaler_obs.s_mu) / scaler_obs.s_sd)
    tr_e = np.arange(CFG["n_exp_train"])
    te_e = np.arange(CFG["n_exp_train"], n_exp)
    X_tl = to_tensor(scaler_in(Xexp_all[tr_e]))
    S_tl = S_exp_std[tr_e]
    # 逆方差加权：把物理单位下的测量噪声换算到标准化空间
    obs_sigma_std = MEAS_SIGMA / scaler_obs.s_sd
    t0 = time.time()
    ens = deep_ensemble_finetune(model, X_tl, S_tl, n_members=CFG["n_members"],
                                 epochs=CFG["tl_epochs"], lr=2e-3, l2sp=3e-3,
                                 obs_sigma=obs_sigma_std, seed=CFG["seed"])
    log(f"     深度集成 {len(ens)} 个成员，用时 {time.time()-t0:.1f}s")

    # 盲测对比
    X_te_e = to_tensor(scaler_in(Xexp_all[te_e]))
    pred_sim_only = predict_scalars(model, X_te_e, scaler_obs)
    pred_tl = ensemble_predict_scalars(ens, X_te_e, scaler_obs)["mean"]
    obs_te_e = exp_S[te_e]
    err_sim = metrics.mixed_error(pred_sim_only, obs_te_e)
    err_tl = metrics.mixed_error(pred_tl, obs_te_e)
    log("\n     盲测（" + str(len(te_e)) + " 发未见实验）平均相对误差：")
    log(metrics.table(err_sim, err_tl))

    # ------------------------------------------------------------ S5 贝叶斯标定
    log("\n[S5] 贝叶斯标定 + 不确定度传播（对应 Spears et al. 2025）")
    # 残差必须用**样本外**的发次来算：如果用参与微调的那 25 发，
    # 代理已经拟合过它们，残差会偏小，涨落幅度 s 会被系统性低估。
    J = bayes.jacobian_logy(ens, exp_designs[te_e], scaler_in, scaler_obs)
    f_ho = ensemble_predict_scalars(ens, X_te_e, scaler_obs)["mean"][:, bayes.IDX_LOGY]
    residual = exp_S[te_e, bayes.IDX_LOGY] - f_ho
    post = bayes.calibrate_variability(J, residual, seed=CFG["seed"])
    log(f"     MCMC 接受率 {post['accept_rate']:.2f}；"
        f"涨落幅度 s = {np.median(post['s']):.2f} (中位数) "
        f"[{np.quantile(post['s'],0.05):.2f}, {np.quantile(post['s'],0.95):.2f}] 90%CI")
    log(f"     残余系统偏移 μ = {np.median(post['mu']):+.3f} (log10 产额)")

    thr_new = float(NEW_DESIGN[0])
    res_new = bayes.propagate(ens, NEW_DESIGN, post, scaler_in, scaler_obs,
                              thr_new, n_mc=384, n_post=160, seed=11)
    res_prev = bayes.propagate(ens, PREV_DESIGN, post, scaler_in, scaler_obs,
                               float(PREV_DESIGN[0]), n_mc=384, n_post=160, seed=12)
    tp_prev, tm_prev = true_p_ignition(PREV_DESIGN, float(PREV_DESIGN[0]), seed=7100)
    tp_new, tm_new = true_p_ignition(NEW_DESIGN, float(NEW_DESIGN[0]), seed=7200)
    log(f"\n     上一代设计 (E={PREV_DESIGN[0]:.2f} MJ): "
        f"P(点火)={float(res_prev['p_ignition'][0])*100:5.1f}%  (真值 {tp_prev*100:5.1f}%)  "
        f"产额中位数 {res_prev['yield_p50_MJ'][0]:.2f} MJ (真值 {tm_prev:.2f}) "
        f"[{res_prev['yield_p05_MJ'][0]:.2f}, {res_prev['yield_p95_MJ'][0]:.2f}] 90%CI")
    log(f"     新一代设计 (E={NEW_DESIGN[0]:.2f} MJ): "
        f"P(点火)={float(res_new['p_ignition'][0])*100:5.1f}%  (真值 {tp_new*100:5.1f}%)  "
        f"产额中位数 {res_new['yield_p50_MJ'][0]:.2f} MJ (真值 {tm_new:.2f}) "
        f"[{res_new['yield_p05_MJ'][0]:.2f}, {res_new['yield_p95_MJ'][0]:.2f}] 90%CI")

    # ------------------------------------------------------------ S6 逆向设计
    log("\n[S6] 稳健性逆向设计：最大化 P(Y>thr) 而非 E[Y]")
    log(f"     硬约束：激光能量固定为 {CFG['energy_budget_mj']:.2f} MJ（设施上限）；"
        f"任务要求产额 > {CFG['mission_threshold_mj']:.2f} MJ（增益 > 1）")
    res_des = dsign.optimize(ens, post, scaler_in, scaler_obs,
                             CFG["mission_threshold_mj"],
                             fixed={0: CFG["energy_budget_mj"]},
                             trust_ref_designs=Xte[:, :N_DESIGN],
                             trust_quantile=0.50, margin=0.06,
                             center=NEW_DESIGN, radius=0.22, seed=CFG["seed"])
    for tag in ("naive", "robust"):
        log(f"     {tag:<6}: P(点火)={res_des[tag]['p_ignition']*100:5.1f}%  "
            f"产额中位数 {res_des[tag]['yield_p50']:.2f} MJ  "
            f"90%CI [{res_des[tag]['yield_p05']:.2f}, {res_des[tag]['yield_p95']:.2f}]")
    log("     两个最优设计的关键差异（粗糙度 / 不对称）：")
    log(f"       max E[Y]    rough={res_des['naive_design'][8]:6.1f} nm  "
        f"p2={res_des['naive_design'][7]:+5.2f}%  dR_abl={res_des['naive_design'][3]:.1f} µm")
    log(f"       max P(ign)  rough={res_des['robust_design'][8]:6.1f} nm  "
        f"p2={res_des['robust_design'][7]:+5.2f}%  dR_abl={res_des['robust_design'][3]:.1f} µm")

    # ---- S6b 闭环校验：把推荐设计丢回真实"求解器"复核（代理不能自证）
    log("\n[S6b] 闭环校验：用真实仿真器复核代理推荐的设计")
    verify = {}
    for tag in ("naive", "robust"):
        x = res_des[f"{tag}_design"]
        ys = []
        for m in range(600):
            dlt = sample_perturbations(1, PERTURB_SCALE_PRIOR, np.random.default_rng(9000 + m))[0]
            s, _ = simulate(x, dlt, np.random.default_rng(9000 + m), mode="exp")
            ys.append(s[bayes.IDX_LOGY])
        ys = np.asarray(ys)
        mj = logYn_to_MJ(ys)
        verify[tag] = {
            "p_ign_true": float((mj > CFG["mission_threshold_mj"]).mean()),
            "median_mj": float(np.median(mj)),
            "pred_p": res_des[tag]["p_ignition"],
            "pred_median": res_des[tag]["yield_p50"],
        }
        log(f"     {tag:<6}: 代理预测 P={res_des[tag]['p_ignition']*100:5.1f}% / "
            f"中位数 {res_des[tag]['yield_p50']:5.2f} MJ   →   "
            f"真实复核 P={verify[tag]['p_ign_true']*100:5.1f}% / "
            f"中位数 {verify[tag]['median_mj']:5.2f} MJ")

    # ------------------------------------------------------------ 报告
    log("\n[S7] 生成报告")
    k = int(np.argmin(np.abs(Xte[:, 0] - 2.05)))
    rec = model.predict_obs(Xte_t[k:k + 1]).detach().numpy()
    figs = [
        (report.fig_surrogate(pred_te, Ste, os.path.join(figs_dir, "surrogate.png")),
         "S3: 代理模型在未见仿真集上的精度"),
        (report.fig_images(Ite[k], scaler_obs.decode_images(rec[0, len(SCALAR_NAMES):]),
                           os.path.join(figs_dir, "images.png")),
         "S3: 多模态成像 —— 仿真真值 vs 代理重建"),
        (report.fig_transfer(err_sim, err_tl, os.path.join(figs_dir, "transfer.png")),
         "S4: 迁移学习把仿真-实验鸿沟压下去"),
        (report.fig_ignition(res_new, res_prev, os.path.join(figs_dir, "ignition.png"),
                             "new design", "previous design"),
         "S5: 不确定性传播后的产额预测分布与点火概率"),
        (report.fig_design(res_des, os.path.join(figs_dir, "design.png")),
         "S6: 稳健设计 vs 传统设计"),
        (report.fig_design_table(res_des, os.path.join(figs_dir, "design_table.png")),
         "S6: 两个最优设计的参数对比"),
    ]
    blocks = [
        ("流水线概览", f"""<pre>
S1 仿真数据库      {n_sim} 次内爆仿真（LHS 采样 + 并行执行）
S2 历史实验        {n_exp} 发（{CFG['n_exp_train']} 训练 / {n_exp-CFG['n_exp_train']} 盲测）
S3 代理模型        MaCC: 自编码器(流形) + 前向 + 逆向，潜变量 32 维
S4 迁移学习        深度集成 {len(ens)} 成员，仅重训前向网络末层 + L2-SP 正则
S5 贝叶斯标定      MCMC 推断涨落幅度 s 与残余偏移 μ，蒙特卡洛传播
S6 逆向设计        max E[Y] vs max P(Y&gt;{CFG['mission_threshold_mj']:.2f} MJ)
</pre>"""),
        ("代理模型精度与加速比", f"""<pre>
{chr(10).join(f'{n:<14} err={e:8.3f}{" [dex]" if n.startswith("log10") else " [%]":<7} R2={r:6.3f}' for n, e, r in zip(SCALAR_NAMES, err_surrogate, r2_surrogate))}

单次评估耗时：仿真 {sim_t_per_run*1e3:.3f} ms  vs  代理 {sur_t_per_run*1e6:.3f} µs
加速比 ≈ {speedup:.0f}x   —— 这正是"用代理替代求解器"做大规模 UQ 与优化的前提
</pre>"""),
        ("迁移学习：闭合仿真-实验鸿沟", f"""<pre>
{metrics.table(err_sim, err_tl)}
</pre>
<p>仿真-only 模型的误差来自<strong>系统性偏差</strong>（辐射流体程序的简化假设），
不是噪声，所以再多的仿真数据也消不掉。迁移学习用 {CFG['n_exp_train']} 发真实实验
重训前向网络的最后几层，学的正是"仿真与实验的差值"。</p>"""),
        ("点火概率预测", f"""<pre>
上一代设计  E={PREV_DESIGN[0]:.2f} MJ : P(点火) = {float(res_prev['p_ignition'][0])*100:5.1f}%   (真值 {tp_prev*100:5.1f}%)
                        产额中位数 {res_prev['yield_p50_MJ'][0]:.2f} MJ      (真值 {tm_prev:.2f})
                        90% 置信区间 [{res_prev['yield_p05_MJ'][0]:.2f}, {res_prev['yield_p95_MJ'][0]:.2f}] MJ

新一代设计  E={NEW_DESIGN[0]:.2f} MJ : P(点火) = {float(res_new['p_ignition'][0])*100:5.1f}%   (真值 {tp_new*100:5.1f}%)
                        产额中位数 {res_new['yield_p50_MJ'][0]:.2f} MJ      (真值 {tm_new:.2f})
                        90% 置信区间 [{res_new['yield_p05_MJ'][0]:.2f}, {res_new['yield_p95_MJ'][0]:.2f}] MJ

贝叶斯标定结果：涨落幅度 s = {np.median(post['s']):.2f} × 工程先验
                残余系统偏移 μ = {np.median(post['mu']):+.3f} (log10 产额)
</pre>
<p>这里给出的不是"产额会是多少"，而是"<strong>点火的概率是多少</strong>"——
把打靶涨落、模型认知不确定度和残余偏差一起传播过代理模型得到的分布。</p>"""),
        ("稳健性设计优化", f"""<pre>
硬约束：激光能量 = {CFG['energy_budget_mj']:.2f} MJ（设施上限，钉死）
任务要求：产额 &gt; {CFG['mission_threshold_mj']:.2f} MJ（NAS 点火判据：增益 &gt; 1）

传统目标 max E[Y]      : P(点火) = {res_des['naive']['p_ignition']*100:5.1f}%   产额中位数 {res_des['naive']['yield_p50']:.2f} MJ
                        粗糙度 {res_des['naive_design'][8]:.0f} nm, P2 {res_des['naive_design'][7]:+.2f}%, 烧蚀层厚 {res_des['naive_design'][3]:.1f} µm

稳健目标 max P(Y&gt;thr)  : P(点火) = {res_des['robust']['p_ignition']*100:5.1f}%   产额中位数 {res_des['robust']['yield_p50']:.2f} MJ
                        粗糙度 {res_des['robust_design'][8]:.0f} nm, P2 {res_des['robust_design'][7]:+.2f}%, 烧蚀层厚 {res_des['robust_design'][3]:.1f} µm
</pre>
<p>平均产额最高的设计往往骑在<strong>点火悬崖</strong>边上：涨落一来就掉下去。
把优化目标换成 P(Y&gt;阈值) 之后，模型会自动把预算花在"降低涨落敏感度"上
（更光滑的靶丸、更好的对称性），而不是一味追求峰值性能。</p>
<h3>闭环校验：把推荐设计丢回真实求解器复核</h3>
<pre>
代理不能自证 —— 每个推荐设计都用真实"辐射流体程序"重跑 600 次（含打靶涨落）：

max E[Y]    代理预测 P={verify['naive']['pred_p']*100:5.1f}% / 中位数 {verify['naive']['pred_median']:5.2f} MJ
            真实复核 P={verify['naive']['p_ign_true']*100:5.1f}% / 中位数 {verify['naive']['median_mj']:5.2f} MJ

max P(ign)  代理预测 P={verify['robust']['pred_p']*100:5.1f}% / 中位数 {verify['robust']['pred_median']:5.2f} MJ
            真实复核 P={verify['robust']['p_ign_true']*100:5.1f}% / 中位数 {verify['robust']['median_mj']:5.2f} MJ
</pre>
<p>若两者差距明显，说明代理在训练数据稀疏的区域发生了<strong>外推</strong>，
这时需要扩样（LLNL 用 1.5×10^5 甚至 10^8 量级的仿真来避免这个问题）。</p>"""),
    ]
    html = report.build_html(os.path.join(OUT, "report.html"), blocks, figs)
    log(f"     报告 -> {html}")
    log(f"\n全部完成，总用时 {time.time()-t_all:.1f}s")
    return dict(err_sim=err_sim, err_tl=err_tl, post=post, res_new=res_new,
                res_prev=res_prev, res_des=res_des)


if __name__ == "__main__":
    main()
