"""
miniJAG —— 半解析 ICF 内爆模型（LLNL `HYDRA/HyPyD` 的本地替身）

作用
----
在 CogSim 工作流里，"高保真辐射流体程序"负责产生训练用的大规模仿真数据库。
本模块用一套**有物理结构的降阶模型**复刻同样的输入输出契约：

    输入  设计参数 x (9 维) + 打靶涨落 δ (4 维)
    输出  6 个标量观测量 + 4 张 32x32 的成像图（多模态）

物理链条（每个环节都是文献中的标准标度律，整体做了常数标定）：

    激光功率 ──[黑腔能量平衡 / Stefan-Boltzmann]──> 辐射温度 T_r
    T_r     ──[烧蚀压标度 P_a ~ T_r^3.5]──────────> 烧蚀压 P_a
    P_a, ṁ  ──[火箭模型 v = u ln(m_i/m_f)]────────> 内爆速度 v_imp
    v_imp,α ──[压缩标度]────────────────────────> 面密度 ρR、离子温度 T_i
    ρR, T_i ──[燃烧份额 θ = ρR/(ρR+H_B(T))]──────> 无 α 加热产额 Y_0
    ρR_hs   ──[α 粒子沉积 f_α, 放大 A = 1/(1-f)]──> 点火悬崖 → 最终产额 Y
    不对称/混合 ──[指数退化因子]────────────────> 产额、ρR_hs 惩罚

**注意**：这是教学/方法验证用的降阶模型，不是高保真程序。
常数（K_*）均按 NIF 标称工况（2.05 MJ / 440 TW / R=1050 µm 等）标定，
使标称点落在 v_imp≈400 km/s、ρR≈1.3 g/cm²、Y≈3 MJ 的真实量级上。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 常量与标度

IMG = 32                       # 成像图边长
N_IMG = 4                      # 3 张中子成像 + 1 张 X 射线自发射成像
N_SCALAR = 6
SCALAR_NAMES = [
    "log10_Yn",       # log10(DT 中子产额) —— LLNL 口径（避免相对误差在 0 附近失真）
    "bang_time_ns",
    "burn_width_ps",
    "Tion_keV",
    "dsr",            # 下散射比
    "rhoR_g_cm2",     # 燃料面密度
]

# 各观测量的测量噪声（实验数据的 1σ），用于迁移学习的逆方差加权
MEAS_SIGMA = np.array([0.035, 0.040, 6.0, 0.25, 0.0025, 0.030], dtype=np.float64)

DESIGN_NAMES = [
    "E_laser_MJ", "P_peak_TW", "R_out_um", "dR_abl_um", "dR_ice_um",
    "dopant_atpct", "gas_fill_mgcc", "p2_amp_pct", "rough_nm",
]
PERTURB_NAMES = ["dE_pct", "dP2_pct", "dRough_pct", "dGas_mgcc"]

N_DESIGN = len(DESIGN_NAMES)   # 9
N_PERTURB = len(PERTURB_NAMES) # 4
N_INPUT = N_DESIGN + N_PERTURB # 13

# 设计空间上下界（CogSim 的"设计空间"定义）
DESIGN_BOUNDS = np.array([
    [1.40, 2.20],     # E_laser  [MJ]
    [380.0, 560.0],   # P_peak   [TW]
    [880.0, 1120.0],  # R_out    [µm]
    [55.0, 115.0],    # dR_abl   [µm]
    [55.0, 105.0],    # dR_ice   [µm]
    [0.10, 1.20],     # dopant   [at.%]
    [0.30, 1.20],     # gas_fill [mg/cc]
    [-5.0, 5.0],      # p2_amp   [%]
    [20.0, 300.0],    # rough    [nm]
], dtype=np.float64)

# 打靶涨落先验（工程先验：激光能量 1%、P2 抖动 1.5%、粗糙度 30%、充气 0.05）
PERTURB_SCALE_PRIOR = np.array([0.010, 1.50, 0.300, 0.050], dtype=np.float64)

# 标定常数（见模块 docstring）
SB = 5.670374419e-8        # Stefan-Boltzmann 常数 [W/m^2/K^4]
EV_TO_K = 11604.518
K_TR = 80.8                # T_r[eV] = K_TR * (η·P_TW / A_eff_cm2)^(1/4)
K_PA = 120.0               # 烧蚀压 [Mbar] @ T_r = 300 eV
K_MDOT = 8.3e5             # 质量烧蚀率 [g/s] @ 标称（使剩余壳层远离质量下限）
K_V = 1.096                # 火箭模型收敛/几何修正（标称 v_imp ≈ 400 km/s）
K_RR = 3.17                # 燃料面密度标定
K_TI = 3.2                 # 无 α 加热离子温度 [keV] @ 标称
K_HB = 250.0               # H_B(T) = K_HB*(T/4.5)^-2  [g/cm^2]
K_HS = 0.512               # 热斑面密度标定 [g/cm^2]
RHO_LAMBDA_ALPHA = 0.015   # α 粒子在热斑中的射程 [g/cm^2]
THETA_MAX = 0.08           # 燃耗份额物理上限（点火发次约 5%，上限取 8%）
ALPHA_RANGE_CLIP = 0.985   # 放大因子截断（点火悬崖不发散）
RHO_HDC = 3.5              # HDC 烧蚀层密度 [g/cc]
RHO_DT = 0.25              # DT 冰密度 [g/cc]
E_PER_REACTION_J = 17.6e6 * 1.602176634e-19   # 每次 DT 反应释放能量 [J]
MOLAR_DT = 5.0             # DT 摩尔质量 [g/mol]
NA = 6.02214076e23

def logYn_to_MJ(log_y_n):
    """log10(中子数) → 聚变产能 [MJ]。"""
    return 10.0 ** np.asarray(log_y_n, dtype=np.float64) * E_PER_REACTION_J / 1e6


def MJ_to_logYn(mj):
    """聚变产能 [MJ] → log10(中子数)。"""
    return np.log10(np.maximum(np.asarray(mj, dtype=np.float64), 1e-12) * 1e6
                    / E_PER_REACTION_J)


_XX, _YY = np.meshgrid(np.linspace(-1.0, 1.0, IMG), np.linspace(-1.0, 1.0, IMG), indexing="xy")


# ---------------------------------------------------------------- 物理链条

def radiation_temperature(E_laser, P_peak, gas_fill, R_out):
    """黑腔能量平衡 → 辐射驱动温度 T_r [eV]。

    T_r ∝ (η P / A_eff)^{1/4}（Stefan-Boltzmann）。
    η 为激光→X 光转换效率，随充气密度升高因背散而下降。
    """
    eta = 0.88 - 0.075 * (gas_fill - 0.30)          # 激光-X 光转换效率
    eta = float(np.clip(eta, 0.55, 0.92))
    a_eff = 1.90 * (1.0 + 0.05 * (gas_fill - 0.30))  # 有效辐射面积 [cm^2]
    flux_term = eta * P_peak / a_eff                  # TW/cm^2
    return K_TR * max(flux_term, 1e-6) ** 0.25


def ablation_pressure(T_r, dopant):
    """烧蚀压 P_a [Mbar]，标度 P_a ∝ T_r^3.5，掺杂提高吸收率。"""
    return K_PA * (T_r / 300.0) ** 3.5 * (1.0 + 0.05 * np.sqrt(max(dopant, 0.0)))


def shell_masses(R_out, dR_abl, dR_ice):
    """烧蚀层质量与 DT 冰质量 [g]。"""
    r_out_cm = R_out * 1e-4
    r_abl_mid = (R_out - 0.5 * dR_abl) * 1e-4
    r_ice_mid = (R_out - dR_abl - 0.5 * dR_ice) * 1e-4
    v_abl = 4.0 * np.pi * r_abl_mid ** 2 * (dR_abl * 1e-4)
    v_ice = 4.0 * np.pi * r_ice_mid ** 2 * (dR_ice * 1e-4)
    _ = r_out_cm
    return RHO_HDC * v_abl, RHO_DT * v_ice


def implosion_velocity(P_a_Mbar, mdot_gs, t_accel_s, m_abl, m_ice, R_out):
    """火箭模型：v = u·ln(m_i/m_f)，u = P_a·A/ṁ 为排气速度。"""
    area_m2 = 4.0 * np.pi * (R_out * 1e-6) ** 2
    force_n = P_a_Mbar * 1e11 * area_m2
    mdot_kgs = max(mdot_gs * 1e-3, 1e-6)
    u = force_n / mdot_kgs                       # [m/s]
    m_i = m_abl + m_ice
    dm = mdot_gs * t_accel_s
    m_f = max(m_i - dm, 0.05 * m_i)
    v = K_V * u * np.log(m_i / m_f)              # [m/s]
    return v / 1e3                                # [km/s]


def adiabat(dopant, P_peak, dR_abl):
    """飞行中子（in-flight adiabat）α：掺杂与高峰值功率抬升 α。"""
    a = 2.0 * (1.0 + 0.60 * (dopant - 0.30)) * (P_peak / 440.0) ** 0.5
    a *= (80.0 / dR_abl) ** 0.15
    return float(np.clip(a, 0.8, 6.0))


# ---------------------------------------------------------------- 成像

def _make_images(hs_radius, p2_total, rough_nm, rng):
    """生成 4 张 32x32 成像（3 张中子 + 1 张 X 射线自发射）。

    与 JAG/NIF 数据契约一致：图像归一化到均值 1，
    绝对幅度信息由标量（log10_Y）承载。
    """
    imgs = np.empty((N_IMG, IMG, IMG), dtype=np.float32)
    rough_f = float(np.clip(rough_nm / 300.0, 0.0, 2.0))
    eps2 = 0.45 * (p2_total / 5.0)
    v_off = rng.normal(0.0, 0.06, size=2) + 0.03 * (p2_total / 5.0)

    for k in range(3):
        ang = k * np.pi / 3.0 + rng.uniform(-0.25, 0.25)
        ca, sa = np.cos(ang), np.sin(ang)
        u = ca * _XX + sa * _YY
        v = -sa * _XX + ca * _YY
        eps = eps2 + rng.normal(0.0, 0.02)
        dist = (u / (1.0 + eps)) ** 2 + (v * (1.0 + eps)) ** 2
        rr2 = (_XX - v_off[0]) ** 2 + (_YY - v_off[1]) ** 2
        g = np.exp(-(rr2 + 0.35 * dist) / (2.0 * hs_radius ** 2))
        g *= 1.0 + 0.30 * rough_f * rng.normal(size=g.shape)
        imgs[k] = np.clip(g, 0.0, None)

    eps = 0.50 * (p2_total / 5.0)
    u = _XX / (1.0 + eps)
    v = _YY * (1.0 + eps)
    r = np.sqrt(u ** 2 + v ** 2)
    shell = np.exp(-((r - 0.62) ** 2) / (2.0 * 0.16 ** 2))
    core = 1.4 * np.exp(-(u ** 2 + v ** 2) / (2.0 * 0.30 ** 2))
    g = shell + core
    g *= 1.0 + 0.20 * rough_f * rng.normal(size=g.shape)
    imgs[3] = np.clip(g, 0.0, None)

    for k in range(N_IMG):
        m = imgs[k].mean()
        imgs[k] /= (m + 1e-8)
    return imgs


# ---------------------------------------------------------------- 主仿真

def simulate(design: np.ndarray, perturb: np.ndarray, rng: np.random.Generator,
             mode: str = "sim", return_diag: bool = False):
    """跑一次"内爆"，返回 (scalars[6], images[4,32,32])。

    mode="sim" : 仿真输出（仅含数值噪声）
    mode="exp" : 实验输出（含**系统性模型偏差** + 测量噪声）
                 —— 这个偏差就是 CogSim 要靠迁移学习修正的"仿真-实验鸿沟"。
    """
    d = np.asarray(design, dtype=np.float64)
    p = np.asarray(perturb, dtype=np.float64)

    # ---- 1. 打靶涨落作用后的"实际"参数
    E_laser = d[0] * (1.0 + p[0])
    P_peak = d[1]
    R_out = d[2]
    dR_abl = d[3]
    dR_ice = d[4]
    dopant = d[5]
    gas_fill = max(d[6] + p[3], 0.10)
    p2_total = d[7] + p[1]
    rough = max(d[8] * (1.0 + p[2]), 5.0)

    # ---- 2. 黑腔驱动
    T_r = radiation_temperature(E_laser, P_peak, gas_fill, R_out)

    # ---- 3. 烧蚀与火箭模型
    P_a = ablation_pressure(T_r, dopant)
    t_pulse_ns = E_laser * 1e6 / (P_peak * 1e12) * 1e9     # E[J]/P[W] → ns
    t_accel_s = 0.80 * t_pulse_ns * 1e-9
    mdot = K_MDOT * (T_r / 300.0) ** 2.5 * (R_out / 1050.0) ** 2
    m_abl, m_ice = shell_masses(R_out, dR_abl, dR_ice)
    v_imp = implosion_velocity(P_a, mdot, t_accel_s, m_abl, m_ice, R_out)
    alpha = adiabat(dopant, P_peak, dR_abl)

    # ---- 4. 退化因子：低模不对称 + 表面粗糙度混合 + 飞行中展弦比(RT 不稳定)
    #      薄壳内爆速度快，但飞行中展弦比 IFAR 越大越容易被 Rayleigh-Taylor 撕裂，
    #      这条约束在真实 ICF 设计里是硬约束，没有它优化器会一味追求薄壳高速度。
    f_p2 = np.exp(-0.5 * (p2_total / 4.0) ** 2)
    f_mix = np.exp(-3.0 * (rough / 300.0) ** 1.5 * (1.0 + 0.5 * max(dopant - 0.3, 0.0)))
    ifar = 3.5 * (R_out / dR_abl)                       # 飞行中展弦比（含壳层压缩）
    f_rt = np.exp(-max(0.0, (ifar - 48.0) / 10.0) ** 2)
    degrade = f_p2 * f_mix * f_rt

    # ---- 5. 压缩：面密度、离子温度
    m_fuel_mg = m_ice * 1e3
    rhoR = K_RR * m_fuel_mg ** (1.0 / 3.0) * (v_imp / 400.0) ** 1.2 / alpha ** 0.55
    rhoR *= 0.55 + 0.45 * degrade
    rhoR = float(np.clip(rhoR, 0.05, 6.0))
    T_i0 = K_TI * (v_imp / 400.0) ** 1.6 * (2.0 / alpha) ** 0.15

    # ---- 6. 热斑面密度 → α 粒子沉积 → 点火悬崖
    rhoR_hs = K_HS * (v_imp / 400.0) ** 2.0 * m_fuel_mg ** (1.0 / 3.0) / alpha ** 0.30
    rhoR_hs *= degrade
    rhoR_hs = float(np.clip(rhoR_hs, 1e-3, 1.0))
    f_alpha = rhoR_hs / (rhoR_hs + RHO_LAMBDA_ALPHA)
    f_alpha = min(f_alpha, ALPHA_RANGE_CLIP)
    amp = 1.0 / (1.0 - f_alpha)                      # α 加热放大倍数

    # ---- 7. 燃烧份额与产额
    H_B = K_HB * (max(T_i0, 0.5) / 4.5) ** -2.0      # H_B ∝ 1/<σv>
    theta_no_alpha = rhoR / (rhoR + H_B)
    # 燃耗份额上限：真实 ICF 的燃耗份额最多几个百分点（NIF 点火发次约 5%）；
    # 没有这个物理天花板，高内爆速度区会算出不真实的"增益十几倍"。
    theta = min(theta_no_alpha * amp, THETA_MAX)
    n_pairs = m_ice / MOLAR_DT * NA
    Yn = max(theta * n_pairs, 1.0)            # DT 中子数
    Y_MJ = Yn * E_PER_REACTION_J / 1e6

    T_ion = T_i0 * (1.0 + 0.60 * np.log(max(amp, 1.0)))

    # ---- 8. 其它观测量
    t_flight_ns = (R_out * 1e-6) / (max(v_imp, 1.0) * 1e3) * 1e9
    bang_time = 1.60 + t_pulse_ns + 0.85 * t_flight_ns
    burn_width = 150.0 * (m_fuel_mg / 0.217) ** (1.0 / 3.0) * (T_ion / 8.7) ** -0.5
    burn_width *= 1.0 + 0.02 * abs(p2_total)
    dsr = 0.032 * (rhoR / 1.30)

    scalars = np.array([
        np.log10(Yn),
        bang_time,
        burn_width,
        T_ion,
        dsr,
        rhoR,
    ], dtype=np.float64)

    # ---- 9. 成像
    hs_radius = float(np.clip(0.34 * (rhoR / 1.30) ** (-1.0 / 3.0) * (1.0 + 0.25 * abs(p2_total) / 5.0),
                              0.10, 0.60))
    images = _make_images(hs_radius, p2_total, rough, rng)

    # ---- 10. 噪声与"仿真-实验"系统性偏差
    if mode == "exp":
        # 系统性偏差：仿真系统性高估产额，且粗糙度越大偏差越大
        b = np.array([
            -0.35 - 0.90 * (rough / 300.0) + 0.25 * (dopant / 1.20),
            +0.25 + 0.10 * (P_peak / 440.0 - 1.0),
            +18.0,
            -0.80 - 0.30 * (rough / 300.0),
            -0.004,
            -0.080,
        ])
        scalars = scalars + b
        noise = np.asarray(rng.normal(0.0, 1.0, size=N_SCALAR), dtype=np.float64) * MEAS_SIGMA
        scalars = scalars + noise
    else:
        scalars = scalars + rng.normal(0.0, 0.004, size=N_SCALAR)

    if return_diag:
        diag = dict(T_rad_eV=T_r, P_a_Mbar=P_a, t_pulse_ns=t_pulse_ns,
                    mdot_gs=mdot, m_abl_g=m_abl, m_ice_g=m_ice,
                    v_imp_kms=v_imp, adiabat=alpha, ifar=ifar,
                    rhoR_hs=rhoR_hs, f_alpha=f_alpha, amp=amp,
                    theta=theta, Y_MJ=float(Y_MJ), degrade=float(degrade))
        return scalars.astype(np.float32), images, diag
    return scalars.astype(np.float32), images
