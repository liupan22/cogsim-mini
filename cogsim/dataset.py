"""数据集：多模态观测（标量 + 成像）的标准化与张量化。

JAG/NIF 数据的关键特征是**多模态**——一次打靶同时给出若干标量（产额、bang time、
离子温度、下散射比…）和若干张成像（中子成像、X 射线自发射）。
传统做法把这些压成几个手工特征，会丢掉绝大部分信息；
CogSim 的做法是直接对原始多模态数据建模，这也是 MaCC 代理的出发点。
"""
from __future__ import annotations

import numpy as np
import torch

from .simulator import IMG, N_IMG, N_SCALAR


class ObsScaler:
    """多模态观测的标准化器：标量按列标准化，成像按像素标准化（先 log1p）。"""

    def __init__(self):
        self.s_mu = None
        self.s_sd = None
        self.i_mu = None
        self.i_sd = None

    def fit(self, scalars: np.ndarray, images: np.ndarray):
        self.s_mu = scalars.mean(axis=0).astype(np.float64)
        self.s_sd = (scalars.std(axis=0) + 1e-8).astype(np.float64)
        im = np.log1p(np.maximum(images, 0.0)).astype(np.float32)
        self.i_mu = im.mean(axis=0)
        self.i_sd = (im.std(axis=0) + 1e-8)
        return self

    def encode(self, scalars, images):
        s = (scalars - self.s_mu) / self.s_sd
        im = np.log1p(np.maximum(images, 0.0))
        i = (im - self.i_mu) / self.i_sd
        return np.hstack([s, i.reshape(len(i), -1)]).astype(np.float32)

    def decode_scalars(self, s_norm):
        return s_norm * self.s_sd + self.s_mu

    def decode_images(self, i_norm):
        im = i_norm.reshape(-1, N_IMG, IMG, IMG) * self.i_sd + self.i_mu
        return np.expm1(im)

    @property
    def obs_dim(self):
        return N_SCALAR + N_IMG * IMG * IMG


class ObsView:
    """把扁平化的观测向量拆成 (标量, 成像) 两部分。"""

    def __init__(self, n_scalar: int = N_SCALAR, n_img: int = N_IMG, img: int = IMG):
        self.n_s = n_scalar
        self.n_i = n_img
        self.img = img

    def split(self, obs):
        return obs[:, : self.n_s], obs[:, self.n_s:].reshape(-1, self.n_i, self.img, self.img)


class InputScaler:
    """设计/涨落参数的标准化器。实验数据的输入必须用仿真数据的统计量变换。"""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd).astype(np.float32)


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
