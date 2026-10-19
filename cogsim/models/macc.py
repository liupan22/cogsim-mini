"""MaCC 代理模型：流形一致 + 循环一致的多模态神经网络代理。

对应文献
--------
Anirudh, Thiagarajan, Bremer & Spears (LLNL),
"Improved Surrogates in Inertial Confinement Fusion with Manifold and
Cycle Consistencies", arXiv:1912.08113（JAG 数据集上的 ICF 代理）。

结构（四个网络互联，而不是一个"黑箱回归器"）
------------------------------------------
    x ──[Forward F]──> z ──[Decoder D]──> ô        前向代理：设计参数 → 观测
    o ──[Encoder E]──> z ──[Decoder D]──> ô        自编码器：物理流形
    z ──[Inverse  I]──> x̂                          逆向：由观测反推设计参数

损失函数
--------
    L = L_recon(D(E(o)), o)                      自编码重建
      + λ_m · ||F(x) − E(o)||²                  流形一致（前向输出必须落在物理流形上）
      + λ_c · ||I(F(x)) − x||²                  循环一致（前向再逆向要回到原处）
      + λ_c · ||F(I(E(o))) − E(o)||²            反向循环一致

为什么这样更好（论文的发现）：
  · 预测始终"物理合法"（在流形上），不会给出物理上不可能的观测组合；
  · 对采样假象更鲁棒，小样本区表现更好；
  · 顺便免费得到一个可微的逆向模型，可直接做反演与自适应采样。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..dataset import ObsView


def _mlp(dims, act=nn.ReLU, last_linear=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or not last_linear:
            layers.append(act())
    return nn.Sequential(*layers)


class MaCC(nn.Module):
    def __init__(self, x_dim: int, obs_dim: int, z_dim: int = 32,
                 hidden: int = 512, view: ObsView | None = None):
        super().__init__()
        self.view = view or ObsView()
        self.encoder = _mlp([obs_dim, hidden, 128, z_dim])
        self.decoder = _mlp([z_dim, 128, hidden, obs_dim])
        self.forward_net = _mlp([x_dim, 128, 128, z_dim])
        self.inverse_net = _mlp([z_dim, 128, 128, x_dim])
        self.z_dim = z_dim
        self.x_dim = x_dim

    # ------------------------------------------------------------ 前向代理
    def predict_obs(self, x):
        """设计参数（含打靶涨落）→ 标准化观测向量。"""
        return self.decoder(self.forward_net(x))

    def predict_latent(self, x):
        return self.forward_net(x)

    def encode(self, obs):
        return self.encoder(obs)

    def invert(self, z):
        return self.inverse_net(z)


def train_macc(model: MaCC,
               X: torch.Tensor,
               Obs: torch.Tensor,
               epochs_ae: int = 120,
               epochs_joint: int = 180,
               lr: float = 1e-3,
               weight_scalar: float = 25.0,
               lam_manifold: float = 1.0,
               lam_cycle: float = 1.0,
               batch_size: int = 256,
               seed: int = 0,
               verbose: bool = True) -> MaCC:
    """两阶段训练：先训自编码器定义物理流形，再训前向/逆向网络并联合微调。"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    view = model.view
    n_scalar = view.n_s

    # 标量加权：标量是实验对标的核心量，权重高于像素
    w = torch.ones(Obs.shape[1])
    w[:n_scalar] = weight_scalar

    def batches():
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            yield perm[i:i + batch_size]

    def recon_loss(o_hat, o):
        return (((o_hat - o) ** 2) * w).mean()

    # ---------- 阶段 1：自编码器（定义物理流形） ----------
    opt = torch.optim.Adam(list(model.encoder.parameters()) +
                           list(model.decoder.parameters()), lr=lr)
    model.train()
    for ep in range(epochs_ae):
        tot = 0.0
        for b in batches():
            o = Obs[b]
            z = model.encoder(o)
            loss = recon_loss(model.decoder(z), o)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if verbose and (ep + 1) % max(1, epochs_ae // 4) == 0:
            print(f"  [AE]    epoch {ep+1:4d}/{epochs_ae}  recon={tot/n:.5f}")

    # ---------- 阶段 2：前向 + 逆向（流形一致 + 循环一致） ----------
    opt = torch.optim.Adam(list(model.forward_net.parameters()) +
                           list(model.inverse_net.parameters()) +
                           list(model.encoder.parameters()) +
                           list(model.decoder.parameters()), lr=lr)
    for ep in range(epochs_joint):
        tot = m_ = c_ = r_ = 0.0
        for b in batches():
            x, o = X[b], Obs[b]
            z = model.encoder(o)
            zf = model.forward_net(x)
            loss_r = recon_loss(model.decoder(z), o)
            loss_m = ((zf - z.detach()) ** 2).mean()
            loss_c = ((model.inverse_net(zf) - x) ** 2).mean()
            loss_c2 = ((model.forward_net(model.inverse_net(z.detach())) - z.detach()) ** 2).mean()
            loss = loss_r + lam_manifold * loss_m + lam_cycle * (loss_c + loss_c2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            m_ += loss_m.item() * len(b)
            c_ += (loss_c.item() + loss_c2.item()) * len(b)
            r_ += loss_r.item() * len(b)
        if verbose and (ep + 1) % max(1, epochs_joint // 4) == 0:
            print(f"  [MaCC]  epoch {ep+1:4d}/{epochs_joint}  total={tot/n:.5f} "
                  f"recon={r_/n:.5f} manifold={m_/n:.5f} cycle={c_/n:.5f}")
    return model


@torch.no_grad()
def predict_scalars(model: MaCC, X: np.ndarray | torch.Tensor, scaler, batch: int = 4096):
    """用代理模型预测**物理量纲**的标量（自动反标准化）。"""
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    out = []
    for i in range(0, X.shape[0], batch):
        obs = model.predict_obs(X[i:i + batch])
        out.append(obs[:, : model.view.n_s].cpu().numpy())
    s_norm = np.vstack(out)
    return scaler.decode_scalars(s_norm)
