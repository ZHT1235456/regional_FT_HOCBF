"""
系统动力学模型：双积分器和三阶积分器。

用于验证 HOCBF 的有限时间恢复性质。
"""

import numpy as np
from abc import ABC, abstractmethod


class ControlAffineSystem(ABC):
    """控制仿射系统基类：dx = f(x) + g(x)u"""

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """漂移项 f(x)"""
        pass

    @abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        """控制向量场 g(x)"""
        pass

    @abstractmethod
    def h(self, x: np.ndarray) -> float:
        """安全函数 h(x)，安全集 C = {x : h(x) >= 0}"""
        pass

    @abstractmethod
    def relative_degree(self) -> int:
        """相对度 r"""
        pass

    @abstractmethod
    def state_dim(self) -> int:
        """状态维度 n"""
        pass

    def dynamics(self, x: np.ndarray, u: float) -> np.ndarray:
        """完整动力学 dx/dt = f(x) + g(x)u"""
        return self.f(x) + self.g(x).flatten() * u


class DoubleIntegrator(ControlAffineSystem):
    """
    双积分器：q'' = u
    状态：x = [q, q_dot]
    相对度：r = 2
    安全函数：h(x) = q_max - q
    """

    def __init__(self, q_max: float = 1.0):
        self._q_max = q_max

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[1], 0.0])

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0])

    def h(self, x: np.ndarray) -> float:
        return self._q_max - x[0]

    def relative_degree(self) -> int:
        return 2

    def state_dim(self) -> int:
        return 2

    def Lf_h(self, x: np.ndarray) -> float:
        """L_f h = -q_dot"""
        return -x[1]

    def Lf2_h(self, x: np.ndarray) -> float:
        """L_f^2 h = 0"""
        return 0.0

    def LgLf_h(self, x: np.ndarray) -> float:
        """L_g L_f h = -1"""
        return -1.0


class TripleIntegrator(ControlAffineSystem):
    """
    三阶积分器：q''' = u
    状态：x = [q, q_dot, q_ddot]
    相对度：r = 3
    安全函数：h(x) = q_max - q
    """

    def __init__(self, q_max: float = 1.0):
        self._q_max = q_max

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[1], x[2], 0.0])

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def h(self, x: np.ndarray) -> float:
        return self._q_max - x[0]

    def relative_degree(self) -> int:
        return 3

    def state_dim(self) -> int:
        return 3

    def Lf_h(self, x: np.ndarray) -> float:
        """L_f h = -q_dot"""
        return -x[1]

    def Lf2_h(self, x: np.ndarray) -> float:
        """L_f^2 h = -q_ddot"""
        return -x[2]

    def Lf3_h(self, x: np.ndarray) -> float:
        """L_f^3 h = 0"""
        return 0.0

    def LgLf2_h(self, x: np.ndarray) -> float:
        """L_g L_f^2 h = -1"""
        return -1.0
