"""迁移学习（Transfer Learning）：把仿真训练的代理模型"校正"到实验。

对应文献
--------
Humbird, Peterson, Salmonson & Spears (LLNL),
"Cognitive simulation models for inertial confinement fusion:
Combining simulation and experimental data", Phys. Plasmas 28, 042709 (2021).

核心思想
--------
仿真数据库很大（10^5 量级）但有**系统性偏差**；实验数据极少（几十发）但**真实**。
做法是：
  1. 先在大规模仿真上训练网络，学会 ICF 的物理骨架（输入→输出的结构关系）；
  2. 冻结大部分层，只用少量实验数据重训最后几层，
     让网络"学会仿真与实验之间的差距"（simulation–experiment gap）。

LLNL 的实测结果（该论文）：用 20 发实验重训后，平均相对误差
  产额(log)  73% → 3%
  离子温度  200% → 7%
  bang time  9% → 4%
本模块复现的正是这一步，并额外用**深度集成**（不同随机种子多次微调）
给出认知不确定度（epistemic uncertainty）。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .macc import MaCC


def finetune(model: MaCC,
             X_exp: torch.Tensor,
             S_exp: torch.Tensor,
             epochs: int = 300,
             lr: float = 2e-3,
             l2sp: float = 1e-2,
             n_unfrozen: int = 2,
             obs_sigma: np.ndarray | None = None,
             weight_power: float = 1.0,
             reinit_unfrozen: bool = True,
             seed: int = 0,
             verbose: bool = False) -> MaCC:
    """用少量实验数据微调前向网络。

    - 自编码器（encoder/decoder）**冻结**：物理流形由仿真定义，不能被动摇；
    - 只训练 forward_net 的最后几层；
    - L2-SP 正则：把参数拉回仿真解附近，防止几十个样本上过拟合；
    - 逆方差加权：测量噪声小的观测量（如产额）权重大，噪声大的（如 burn width）权重小。
      少了这条，网络会被噪声大的量带偏。
    """
    torch.manual_seed(seed)
    net = copy.deepcopy(model)
    n_scalar = net.view.n_s
    if obs_sigma is None:
        w_obs = torch.ones(n_scalar)
    else:
        sg = np.asarray(obs_sigma, dtype=np.float64)[:n_scalar]
        w_obs = torch.from_numpy((1.0 / np.maximum(sg, 1e-9)) ** weight_power).float()
        w_obs = w_obs / w_obs.mean()

    # 冻结自编码器
    for p in net.encoder.parameters():
        p.requires_grad_(False)
    for p in net.decoder.parameters():
        p.requires_grad_(False)
    for p in net.inverse_net.parameters():
        p.requires_grad_(False)

    # 只解冻 forward_net 的最后若干个 Linear 层
    n_mod = len(net.forward_net)
    last_linear = (n_mod - 1) // 2
    unfreeze_from = max(0, last_linear - (n_unfrozen - 1))
    params = []
    ref = {}
    for name, p in net.forward_net.named_parameters():
        layer_idx = int(name.split(".")[0]) // 2
        if layer_idx >= unfreeze_from:
            p.requires_grad_(True)
            params.append(p)
            ref[name] = p.detach().clone()
        else:
            p.requires_grad_(False)

    # 对解冻的层做 seed 相关的**重新初始化** —— 深度集成的多样性来源。
    # 不这么做，各成员继承自同一个预训练权重，离散度≈0，
    # "集成离散度 = 认知不确定度"这条假设就失效了，可信域过滤也会形同虚设。
    if reinit_unfrozen:
        g = torch.Generator().manual_seed(seed + 991)
        for name, p in net.forward_net.named_parameters():
            if name in ref:
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, generator=g)
                else:
                    nn.init.zeros_(p)

    opt = torch.optim.Adam(params, lr=lr)
    net.train()
    n = X_exp.shape[0]
    n_scalar = net.view.n_s

    for ep in range(epochs):
        zf = net.forward_net(X_exp)
        obs = net.decoder(zf)
        loss = (((obs[:, :n_scalar] - S_exp) ** 2) * w_obs.to(obs.device)).mean()
        reg = sum(((p - ref[n_]) ** 2).sum() for n_, p in net.forward_net.named_parameters()
                  if n_ in ref)
        total = loss + l2sp * reg
        opt.zero_grad()
        total.backward()
        opt.step()
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    [TL] epoch {ep+1:4d}/{epochs}  mse={loss.item():.5f}")
    return net


def deep_ensemble_finetune(model: MaCC,
                           X_exp: torch.Tensor,
                           S_exp: torch.Tensor,
                           n_members: int = 4,
                           **kw) -> list[MaCC]:
    """深度集成：用不同随机种子微调多个模型，用成员间离散度估计认知不确定度。"""
    out = []
    for k in range(n_members):
        kw_k = dict(kw)
        kw_k["seed"] = kw.get("seed", 0) + 17 * k
        out.append(finetune(model, X_exp, S_exp, **kw_k))
    return out


@torch.no_grad()
def ensemble_predict_scalars(models: list[MaCC], X, scaler, batch: int = 4096) -> dict:
    """集成预测：返回均值、成员标准差（认知不确定度）与全部成员预测。"""
    preds = []
    for m in models:
        m.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    for m in models:
        chunks = []
        for i in range(0, X.shape[0], batch):
            obs = m.predict_obs(X[i:i + batch])
            chunks.append(obs[:, : m.view.n_s].cpu().numpy())
        preds.append(scaler.decode_scalars(np.vstack(chunks)))
    P = np.stack(preds)                       # (M, N, n_scalar)
    return {"mean": P.mean(axis=0), "std": P.std(axis=0), "members": P}
