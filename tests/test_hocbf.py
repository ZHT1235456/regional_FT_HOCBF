"""HOCBF 模块单元测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.systems import DoubleIntegrator, TripleIntegrator
from src.hocbf import compute_psi, compute_psi_r, hocbf_constraint_coefficients
from src.recovery import (
    recovery_polynomial, recovery_polynomial_coeffs,
    check_sufficient_condition, theoretical_time_bound,
    find_polynomial_positive_root, recovery_analysis,
)


class TestDoubleIntegrator:
    """双积分器系统测试"""

    def test_f(self):
        sys = DoubleIntegrator()
        x = np.array([1.0, 2.0])
        np.testing.assert_array_equal(sys.f(x), np.array([2.0, 0.0]))

    def test_g(self):
        sys = DoubleIntegrator()
        x = np.array([1.0, 2.0])
        np.testing.assert_array_equal(sys.g(x), np.array([0.0, 1.0]))

    def test_h(self):
        sys = DoubleIntegrator(q_max=1.0)
        x_safe = np.array([0.5, 0.0])
        x_unsafe = np.array([1.5, 0.0])
        assert sys.h(x_safe) == 0.5
        assert sys.h(x_unsafe) == -0.5

    def test_relative_degree(self):
        sys = DoubleIntegrator()
        assert sys.relative_degree() == 2

    def test_lie_derivatives(self):
        sys = DoubleIntegrator()
        x = np.array([1.0, 2.0])
        assert sys.Lf_h(x) == -2.0
        assert sys.Lf2_h(x) == 0.0
        assert sys.LgLf_h(x) == -1.0


class TestTripleIntegrator:
    """三阶积分器系统测试"""

    def test_f(self):
        sys = TripleIntegrator()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sys.f(x), np.array([2.0, 3.0, 0.0]))

    def test_relative_degree(self):
        sys = TripleIntegrator()
        assert sys.relative_degree() == 3


class TestHOCBF:
    """HOCBF 辅助变量计算测试"""

    def test_psi_double_integrator(self):
        sys = DoubleIntegrator(q_max=1.0)
        x0 = np.array([1.5, -1.0])
        k = 1.0

        psi = compute_psi(sys, x0, k)
        assert len(psi) == 2
        # ψ0 = h = q_max - q = 1.0 - 1.5 = -0.5
        np.testing.assert_almost_equal(psi[0], -0.5)
        # ψ1 = -q_dot + k*h = -(-1) + 1*(-0.5) = 1 - 0.5 = 0.5
        np.testing.assert_almost_equal(psi[1], 0.5)

    def test_psi_triple_integrator(self):
        sys = TripleIntegrator(q_max=1.0)
        x0 = np.array([1.5, -1.0, 0.0])
        k = 1.0

        psi = compute_psi(sys, x0, k)
        assert len(psi) == 3
        np.testing.assert_almost_equal(psi[0], -0.5)
        np.testing.assert_almost_equal(psi[1], 0.5)
        # ψ2 = -q_ddot - 2k*q_dot + k^2*(q_max - q)
        #      = 0 - 2*1*(-1) + 1*(-0.5) = 2 - 0.5 = 1.5
        np.testing.assert_almost_equal(psi[2], 1.5)

    def test_psi_r_double_integrator(self):
        sys = DoubleIntegrator(q_max=1.0)
        x0 = np.array([1.5, -1.0])
        k = 1.0
        u = 0.0

        psi_r = compute_psi_r(sys, x0, u, k)
        # ψ2 = -u - 2k*q_dot + k^2*(q_max - q)
        #     = 0 - 2*1*(-1) + 1*(-0.5) = 2 - 0.5 = 1.5
        np.testing.assert_almost_equal(psi_r, 1.5)

    def test_hocbf_constraint_coefficients(self):
        sys = DoubleIntegrator(q_max=1.0)
        x0 = np.array([1.5, -1.0])
        k = 1.0

        a, b = hocbf_constraint_coefficients(sys, x0, k)
        # a = LgLf_h = -1
        np.testing.assert_almost_equal(a, -1.0)
        # b = Lf2_h + 2k*Lf_h + k^2*h = 0 + 2*1*(-(-1)) + 1*(-0.5) = 2 - 0.5 = 1.5
        np.testing.assert_almost_equal(b, 1.5)


class TestRecovery:
    """恢复多项式和时间界测试"""

    def test_recovery_polynomial_r2(self):
        # P_1(t) = ψ0 + ψ1*t
        psi = np.array([-0.5, 0.5])
        assert recovery_polynomial(psi, 0.0) == -0.5
        assert recovery_polynomial(psi, 1.0) == 0.0
        assert recovery_polynomial(psi, 2.0) == 0.5

    def test_recovery_polynomial_coeffs(self):
        psi = np.array([-0.5, 0.5])
        coeffs = recovery_polynomial_coeffs(psi)
        np.testing.assert_array_almost_equal(coeffs, [-0.5, 0.5])

    def test_sufficient_condition_satisfied(self):
        psi = np.array([-0.5, 0.5])
        satisfied, msg = check_sufficient_condition(psi)
        assert satisfied is True

    def test_sufficient_condition_not_satisfied_h_positive(self):
        psi = np.array([0.5, 0.5])
        satisfied, msg = check_sufficient_condition(psi)
        assert satisfied is False
        assert "已安全" in msg

    def test_sufficient_condition_not_satisfied_psi1_negative(self):
        psi = np.array([-0.5, -0.5])
        satisfied, msg = check_sufficient_condition(psi)
        assert satisfied is False
        assert "<= 0" in msg

    def test_theoretical_time_bound(self):
        psi = np.array([-0.5, 0.5])
        bound = theoretical_time_bound(psi)
        np.testing.assert_almost_equal(bound, 1.0)

    def test_polynomial_root_r2(self):
        psi = np.array([-0.5, 0.5])
        root = find_polynomial_positive_root(psi)
        np.testing.assert_almost_equal(root, 1.0)

    def test_polynomial_root_r3(self):
        # P_2(t) = ψ0 + ψ1*t + ψ2/2*t^2
        # -0.5 + 0.5*t + 0.75*t^2 = 0
        psi = np.array([-0.5, 0.5, 1.5])
        root = find_polynomial_positive_root(psi)
        # 0.75*t^2 + 0.5*t - 0.5 = 0
        # t = (-0.5 + sqrt(0.25 + 1.5)) / 1.5 = (-0.5 + sqrt(1.75)) / 1.5
        expected = (-0.5 + np.sqrt(1.75)) / 1.5
        np.testing.assert_almost_equal(root, expected)

    def test_recovery_analysis(self):
        psi = np.array([-0.5, 0.5])
        result = recovery_analysis(psi)
        assert result['sufficient_condition'] is True
        np.testing.assert_almost_equal(result['theoretical_bound'], 1.0)
        np.testing.assert_almost_equal(result['polynomial_root'], 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
