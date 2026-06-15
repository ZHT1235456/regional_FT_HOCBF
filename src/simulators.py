"""
QP 安全滤波器仿真器。

每步求解：min_u ||u - u_nom||^2  s.t.  ψ_r(x, u) >= 0
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass

from src.systems import ControlAffineSystem
from src.hocbf import hocbf_constraint_coefficients


@dataclass
class SimResult:
    """仿真结果"""
    t: np.ndarray          # 时间序列
    x: np.ndarray          # 状态序列 (n_steps, n_states)
    u: np.ndarray          # 控制输入序列 (n_steps,)
    h: np.ndarray          # 安全函数序列 (n_steps,)
    psi_r: np.ndarray      # ψ_r 序列 (n_steps,)
    recovered: bool        # 是否恢复
    recovery_time: Optional[float]  # 恢复时间


def qp_filter_scalar(a: float, b: float, u_nom: float) -> float:
    """
    标量 QP 安全滤波器。

    min_u (u - u_nom)^2  s.t.  a*u + b >= 0

    Parameters
    ----------
    a : float
        u 的系数
    b : float
        常数项
    u_nom : float
        名义控制输入

    Returns
    -------
    float
        过滤后的控制输入
    """
    if abs(a) < 1e-12:
        # 约束不含 u，直接返回名义值
        return u_nom

    # 约束：u >= -b/a  (若 a > 0)  或  u <= -b/a  (若 a < 0)
    u_min = -b / a

    if a > 0:
        # u >= u_min
        return max(u_nom, u_min)
    else:
        # u <= u_min
        return min(u_nom, u_min)


def simulate_with_hocbf(
    system: ControlAffineSystem,
    x0: np.ndarray,
    k: float,
    u_nominal: Callable[[np.ndarray, float], float],
    T_max: float = 10.0,
    dt: float = 0.01,
    h_threshold: float = 0.0,
) -> SimResult:
    """
    带 HOCBF 约束的仿真。

    Parameters
    ----------
    system : ControlAffineSystem
        系统模型
    x0 : np.ndarray
        初始状态
    k : float
        HOCBF 增益
    u_nominal : Callable
        名义控制器 u_nom(x, t)
    T_max : float
        最大仿真时间
    dt : float
        时间步长
    h_threshold : float
        恢复判定阈值（默认 0）

    Returns
    -------
    SimResult
        仿真结果
    """
    n_steps = int(T_max / dt) + 1
    t_arr = np.zeros(n_steps)
    x_arr = np.zeros((n_steps, system.state_dim()))
    u_arr = np.zeros(n_steps)
    h_arr = np.zeros(n_steps)
    psi_r_arr = np.zeros(n_steps)

    x = x0.copy()
    recovered = False
    recovery_time = None

    for i in range(n_steps):
        t = i * dt
        t_arr[i] = t
        x_arr[i] = x

        # 计算安全函数
        h_val = system.h(x)
        h_arr[i] = h_val

        # 检查是否恢复
        if not recovered and h_val >= h_threshold:
            recovered = True
            recovery_time = t

        # 名义控制
        u_nom = u_nominal(x, t)

        # HOCBF 约束
        a, b = hocbf_constraint_coefficients(system, x, k)
        psi_r_arr[i] = a * u_nom + b  # 记录名义值下的 ψ_r

        # QP 滤波
        if not recovered:
            # 恢复前强制执行 HOCBF 约束
            u = qp_filter_scalar(a, b, u_nom)
        else:
            # 恢复后可选择是否继续约束（这里选择不约束）
            u = u_nom

        u_arr[i] = u

        # 积分一步
        dx = system.dynamics(x, u)
        x = x + dx * dt

    return SimResult(
        t=t_arr,
        x=x_arr,
        u=u_arr,
        h=h_arr,
        psi_r=psi_r_arr,
        recovered=recovered,
        recovery_time=recovery_time,
    )


def simulate_without_constraint(
    system: ControlAffineSystem,
    x0: np.ndarray,
    u_nominal: Callable[[np.ndarray, float], float],
    T_max: float = 10.0,
    dt: float = 0.01,
) -> SimResult:
    """
    无约束仿真（对照组）。

    Parameters
    ----------
    system : ControlAffineSystem
        系统模型
    x0 : np.ndarray
        初始状态
    u_nominal : Callable
        名义控制器 u_nom(x, t)
    T_max : float
        最大仿真时间
    dt : float
        时间步长

    Returns
    -------
    SimResult
        仿真结果
    """
    n_steps = int(T_max / dt) + 1
    t_arr = np.zeros(n_steps)
    x_arr = np.zeros((n_steps, system.state_dim()))
    u_arr = np.zeros(n_steps)
    h_arr = np.zeros(n_steps)

    x = x0.copy()
    recovered = False
    recovery_time = None

    for i in range(n_steps):
        t = i * dt
        t_arr[i] = t
        x_arr[i] = x

        h_val = system.h(x)
        h_arr[i] = h_val

        if not recovered and h_val >= 0:
            recovered = True
            recovery_time = t

        u = u_nominal(x, t)
        u_arr[i] = u

        dx = system.dynamics(x, u)
        x = x + dx * dt

    return SimResult(
        t=t_arr,
        x=x_arr,
        u=u_arr,
        h=h_arr,
        psi_r=np.zeros(n_steps),
        recovered=recovered,
        recovery_time=recovery_time,
    )
