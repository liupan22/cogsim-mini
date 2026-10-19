"""mini-CogSim —— LLNL CogSim 认知仿真框架的本地等比缩小实现。

分层（与 LLNL 一一对应）：
    simulator.py   ←→  HYDRA / HyPyD       高保真辐射流体程序（这里用半解析降阶模型）
    sampling.py    ←→  设计空间采样
    workflow.py    ←→  Merlin              大规模仿真工作流编排
    models/macc.py ←→  MaCC 多模态代理      自编码器 + 前向 + 逆向
    models/transfer.py ←→ 迁移学习          仿真 → 实验校正
    bayes.py       ←→  Bayesian calibration 涨落标定 + 不确定度传播
    design.py      ←→  稳健性设计优化
"""
__version__ = "0.1.0"
