"""
恢复多项式与理论时间界计算。

根据文档：
- 恢复多项式：P_{r-1}(t) = Σ_{i=0}^{r-1} ψ_i(x_0)/i! * t^i
- 定理 1：若 S = {T > 0 : P_{r-1}(T) >= 0} 非空，则 τ_C <= inf S
- 推论 1：若 ψ_1 > 0 且 ψ_i >= 0 (i >= 2)，则 T_rec <= -h(x_0)/ψ_1(x_0)
"""

import numpy as np
from typing import Optional, Tuple


def recovery_polynomial(psi_values: np.ndarray, t: float) -> float:
    """
    计算恢复多项式 P_{r-1}(t)。

    Parameters
    ----------
    psi_values : np.ndarray
        初始辅助变量 [ψ_0(x_0), ψ_1(x_0), ..., ψ_{r-1}(x_0)]
    t : float
        时间

    Returns
    -------
    float
        P_{r-1}(t) 的值
    """
    r = len(psi_values)
    result = 0.0
    factorial = 1.0
    for i in range(r):
        if i > 0:
            factorial *= i
        result += psi_values[i] / factorial * t**i
    return result


def recovery_polynomial_coeffs(psi_values: np.ndarray) -> np.ndarray:
    """
    返回恢复多项式的系数 [a_0, a_1, ..., a_{r-1}]，其中 P(t) = Σ a_i t^i。

    Parameters
    ----------
    psi_values : np.ndarray
        初始辅助变量 [ψ_0(x_0), ψ_1(x_0), ..., ψ_{r-1}(x_0)]

    Returns
    -------
    np.ndarray
        多项式系数
    """
    r = len(psi_values)
    coeffs = np.zeros(r)
    factorial = 1.0
    for i in range(r):
        if i > 0:
            factorial *= i
        coeffs[i] = psi_values[i] / factorial
    return coeffs


def check_sufficient_condition(psi_values: np.ndarray) -> Tuple[bool, str]:
    """
    检查推论 1 的简单充分条件。

    条件：
    - ψ_0(x_0) = h(x_0) < 0（初始不安全）
    - ψ_1(x_0) > 0
    - ψ_i(x_0) >= 0，i = 2, ..., r-1

    Parameters
    ----------
    psi_values : np.ndarray
        初始辅助变量

    Returns
    -------
    satisfied : bool
        是否满足条件
    message : str
        说明信息
    """
    if psi_values[0] >= 0:
        return False, "h(x_0) >= 0，初始状态已安全"

    if len(psi_values) < 2:
        return False, "需要至少 r >= 2"

    if psi_values[1] <= 0:
        return False, f"ψ_1(x_0) = {psi_values[1]:.4f} <= 0"

    for i in range(2, len(psi_values)):
        if psi_values[i] < 0:
            return False, f"ψ_{i}(x_0) = {psi_values[i]:.4f} < 0"

    return True, "满足推论 1 的充分条件"


def theoretical_time_bound(psi_values: np.ndarray) -> Optional[float]:
    """
    计算理论恢复时间界（推论 1）。

    T_rec <= -h(x_0) / ψ_1(x_0)

    Parameters
    ----------
    psi_values : np.ndarray
        初始辅助变量

    Returns
    -------
    float or None
        理论时间界，若条件不满足则返回 None
    """
    satisfied, _ = check_sufficient_condition(psi_values)
    if not satisfied:
        return None
    return -psi_values[0] / psi_values[1]


def find_polynomial_positive_root(psi_values: np.ndarray) -> Optional[float]:
    """
    求恢复多项式 P_{r-1}(t) = 0 的最小正根（定理 1）。

    Parameters
    ----------
    psi_values : np.ndarray
        初始辅助变量

    Returns
    -------
    float or None
        最小正根，若不存在则返回 None
    """
    coeffs = recovery_polynomial_coeffs(psi_values)

    # numpy.roots 求解的是最高次项在前的多项式
    # 需要反转系数顺序
    roots = np.roots(coeffs[::-1])

    # 筛选正实根
    positive_roots = []
    for root in roots:
        if np.isreal(root) and root.real > 0:
            positive_roots.append(root.real)

    if not positive_roots:
        return None

    return min(positive_roots)


def recovery_analysis(psi_values: np.ndarray) -> dict:
    """
    综合恢复分析。

    Parameters
    ----------
    psi_values : np.ndarray
        初始辅助变量 [ψ_0, ψ_1, ..., ψ_{r-1}]

    Returns
    -------
    dict
        分析结果
    """
    result = {
        'psi_values': psi_values,
        'h0': psi_values[0],
        'r': len(psi_values),
    }

    # 检查推论条件
    satisfied, message = check_sufficient_condition(psi_values)
    result['sufficient_condition'] = satisfied
    result['sufficient_message'] = message

    # 理论时间界
    result['theoretical_bound'] = theoretical_time_bound(psi_values)

    # 多项式正根
    result['polynomial_root'] = find_polynomial_positive_root(psi_values)

    return result
