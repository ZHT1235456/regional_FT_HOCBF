"""
HOCBF 辅助变量计算。

根据文档定义：
- ψ_0 = h
- ψ_i = ψ̇_{i-1} + k ψ_{i-1}，i = 1, ..., r

其中 ψ_{r-1} 独立于控制输入，ψ_r 对 u 仿射。
"""

import numpy as np
from src.systems import ControlAffineSystem, DoubleIntegrator, TripleIntegrator


def compute_psi(system: ControlAffineSystem, x: np.ndarray, k: float) -> np.ndarray:
    """
    计算辅助变量 ψ_0, ψ_1, ..., ψ_{r-1}（不含控制输入的项）。

    Parameters
    ----------
    system : ControlAffineSystem
        系统模型
    x : np.ndarray
        当前状态
    k : float
        HOCBF 增益参数

    Returns
    -------
    np.ndarray
        长度为 r 的数组 [ψ_0, ψ_1, ..., ψ_{r-1}]
    """
    r = system.relative_degree()
    psi = np.zeros(r)

    if isinstance(system, DoubleIntegrator):
        # r = 2
        # ψ_0 = h = q_max - q
        psi[0] = system.h(x)
        # ψ_1 = -q_dot + k * (q_max - q)
        psi[1] = system.Lf_h(x) + k * psi[0]

    elif isinstance(system, TripleIntegrator):
        # r = 3
        # ψ_0 = h = q_max - q
        psi[0] = system.h(x)
        # ψ_1 = -q_dot + k * (q_max - q)
        psi[1] = system.Lf_h(x) + k * psi[0]
        # ψ_2 = L_f^2 h + k * L_f h + k^2 * h + k * ψ_1
        #      = -q_ddot + k*(-q_dot) + k*(-q_dot + k*(q_max - q))
        #      = -q_ddot - 2k*q_dot + k^2*(q_max - q)
        psi[2] = system.Lf2_h(x) + 2 * k * system.Lf_h(x) + k**2 * psi[0]

    else:
        # 通用实现（数值计算）
        psi[0] = system.h(x)
        for i in range(1, r):
            # 使用递推关系（需要知道 L_f^i h）
            # 这里使用数值近似，假设系统已提供足够的李导数
            raise NotImplementedError(f"未实现 {type(system).__name__} 的通用 ψ 计算")

    return psi


def compute_psi_r(system: ControlAffineSystem, x: np.ndarray, u: float, k: float) -> float:
    """
    计算 ψ_r（含控制输入）。

    ψ_r = ψ̇_{r-1} + k * ψ_{r-1}

    对于双积分器 (r=2)：
    ψ_2 = -u - 2k*q_dot + k^2*(q_max - q)

    对于三阶积分器 (r=3)：
    ψ_3 = -u + k*(-q_ddot - 2k*q_dot + k^2*(q_max - q))
        = -u - k*q_ddot - 2k^2*q_dot + k^3*(q_max - q)

    Parameters
    ----------
    system : ControlAffineSystem
        系统模型
    x : np.ndarray
        当前状态
    u : float
        控制输入
    k : float
        HOCBF 增益参数

    Returns
    -------
    float
        ψ_r 的值
    """
    r = system.relative_degree()
    psi = compute_psi(system, x, k)

    if isinstance(system, DoubleIntegrator):
        # ψ_2 = L_f^2 h + L_g L_f h * u + C(2,1)*k*L_f h + k^2*h
        #      = 0 + (-1)*u + 2*k*(-q_dot) + k^2*(q_max - q)
        #      = -u - 2k*q_dot + k^2*(q_max - q)
        psi_r = system.Lf2_h(x) + system.LgLf_h(x) * u
        psi_r += 2 * k * system.Lf_h(x) + k**2 * system.h(x)

    elif isinstance(system, TripleIntegrator):
        # ψ_3 = L_f^3 h + L_g L_f^2 h * u + C(3,1)*k*L_f^2 h + C(3,2)*k^2*L_f h + k^3*h
        #      = 0 + (-1)*u + 3*k*(-q_ddot) + 3*k^2*(-q_dot) + k^3*(q_max - q)
        #      = -u - 3k*q_ddot - 3k^2*q_dot + k^3*(q_max - q)
        psi_r = system.Lf3_h(x) + system.LgLf2_h(x) * u
        psi_r += 3 * k * system.Lf2_h(x) + 3 * k**2 * system.Lf_h(x) + k**3 * system.h(x)

    else:
        raise NotImplementedError(f"未实现 {type(system).__name__} 的 ψ_r 计算")

    return psi_r


def hocbf_constraint_coefficients(system: ControlAffineSystem, x: np.ndarray, k: float):
    """
    返回 HOCBF 约束的系数，用于 QP 求解。

    约束形式：ψ_r >= 0
    即：a * u + b >= 0

    Returns
    -------
    a : float
        u 的系数（L_g L_f^{r-1} h）
    b : float
        常数项（不含 u 的部分）
    """
    r = system.relative_degree()

    if isinstance(system, DoubleIntegrator):
        # ψ_2 = -u - 2k*q_dot + k^2*(q_max - q)
        a = system.LgLf_h(x)  # = -1
        b = system.Lf2_h(x) + 2 * k * system.Lf_h(x) + k**2 * system.h(x)

    elif isinstance(system, TripleIntegrator):
        # ψ_3 = -u - 3k*q_ddot - 3k^2*q_dot + k^3*(q_max - q)
        a = system.LgLf2_h(x)  # = -1
        b = (system.Lf3_h(x) + 3 * k * system.Lf2_h(x) +
             3 * k**2 * system.Lf_h(x) + k**3 * system.h(x))

    else:
        raise NotImplementedError(f"未实现 {type(system).__name__} 的约束系数计算")

    return a, b
