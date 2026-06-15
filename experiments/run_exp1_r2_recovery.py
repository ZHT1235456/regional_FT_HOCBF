"""
实验 1：r=2 双积分器有限时间恢复验证。

验证目标：
1. 满足推论条件 (h<0, ψ1>0) 的初始状态能在理论时间界内恢复
2. 不满足条件的初始状态可能无法恢复
3. 数值恢复时间与理论界对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.systems import DoubleIntegrator
from src.hocbf import compute_psi
from src.recovery import recovery_analysis, theoretical_time_bound
from src.simulators import simulate_with_hocbf, simulate_without_constraint


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_exp1():
    """运行实验 1"""
    config = load_config()
    system = DoubleIntegrator(q_max=config['system']['q_max'])
    k = config['hocbf']['k']
    T_max = config['simulation']['T_max']
    dt = config['simulation']['dt']

    # 名义控制器：u_nom = -K * [q - q_ref, q_dot]
    # 使用简单的 PD 控制器，目标是将系统推向安全集
    def u_nominal(x, t):
        q_ref = 0.5  # 目标位置（安全集内部）
        Kp, Kd = 2.0, 2.0
        return -Kp * (x[0] - q_ref) - Kd * x[1]

    # 初始条件
    ics = config['exp1']['initial_conditions']
    results = {}

    print("=" * 60)
    print("实验 1：r=2 双积分器有限时间恢复验证")
    print("=" * 60)

    for name, x0_list in ics.items():
        x0 = np.array(x0_list)
        q0, qdot0 = x0

        # 理论分析
        psi = compute_psi(system, x0, k)
        analysis = recovery_analysis(psi)

        print(f"\n--- 初始条件 {name}: q0={q0}, q_dot0={qdot0} ---")
        print(f"  h(x0) = {analysis['h0']:.4f}")
        print(f"  ψ0 = {psi[0]:.4f}, ψ1 = {psi[1]:.4f}")
        print(f"  推论条件: {analysis['sufficient_message']}")
        if analysis['theoretical_bound'] is not None:
            print(f"  理论恢复时间界: T <= {analysis['theoretical_bound']:.4f}")
        if analysis['polynomial_root'] is not None:
            print(f"  多项式正根: T = {analysis['polynomial_root']:.4f}")

        # 仿真（带 HOCBF）
        sim_hocbf = simulate_with_hocbf(system, x0, k, u_nominal, T_max, dt)

        # 仿真（无约束）
        sim_free = simulate_without_constraint(system, x0, u_nominal, T_max, dt)

        print(f"  [HOCBF] 恢复: {sim_hocbf.recovered}, 恢复时间: {sim_hocbf.recovery_time}")
        print(f"  [无约束] 恢复: {sim_free.recovered}, 恢复时间: {sim_free.recovery_time}")

        if analysis['theoretical_bound'] is not None and sim_hocbf.recovered:
            ratio = sim_hocbf.recovery_time / analysis['theoretical_bound']
            print(f"  数值/理论 比值: {ratio:.4f}")

        results[name] = {
            'x0': x0,
            'psi': psi,
            'analysis': analysis,
            'sim_hocbf': sim_hocbf,
            'sim_free': sim_free,
        }

    # 绘图
    plot_exp1_results(results, k, config)

    return results


def plot_exp1_results(results, k, config):
    """绘制实验 1 结果"""
    project_root = Path(__file__).parent.parent
    save_dir = project_root / config['plot']['save_dir']
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 颜色和标签
    colors = {'A1': 'C0', 'A2': 'C1', 'B1': 'C2', 'C1': 'C3'}
    labels = {
        'A1': r'$q_0=1.5, \dot{q}_0=-1$ ($\psi_1>0$)',
        'A2': r'$q_0=1.5, \dot{q}_0=-0.5$ ($\psi_1>0$)',
        'B1': r'$q_0=1.5, \dot{q}_0=0$ ($\psi_1=0$)',
        'C1': r'$q_0=1.5, \dot{q}_0=1$ ($\psi_1<0$)',
    }

    # 图 1: q(t) 轨迹
    ax = axes[0, 0]
    q_max = config['system']['q_max']
    ax.axhline(y=q_max, color='r', linestyle='--', alpha=0.7, label='Safety boundary')
    for name, data in results.items():
        ax.plot(data['sim_hocbf'].t, data['sim_hocbf'].x[:, 0],
                color=colors[name], label=f"{name} (HOCBF)")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('q(t)')
    ax.set_title('Position Trajectories (with HOCBF)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, config['simulation']['T_max'])

    # 图 2: h(t) 变化
    ax = axes[0, 1]
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='h=0')
    for name, data in results.items():
        ax.plot(data['sim_hocbf'].t, data['sim_hocbf'].h,
                color=colors[name], label=labels.get(name, name))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('h(t) = q_max - q')
    ax.set_title('Safety Function h(t)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图 3: HOCBF vs 无约束对比
    ax = axes[1, 0]
    for name in ['A1', 'C1']:
        if name in results:
            data = results[name]
            ax.plot(data['sim_hocbf'].t, data['sim_hocbf'].h,
                    color=colors[name], label=f"{name} (HOCBF)")
            ax.plot(data['sim_free'].t, data['sim_free'].h,
                    color=colors[name], linestyle=':', label=f"{name} (free)")
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('h(t)')
    ax.set_title('HOCBF vs Unconstrained')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图 4: 恢复时间对比
    ax = axes[1, 1]
    names_with_bound = []
    numerical_times = []
    theoretical_bounds = []
    for name, data in results.items():
        if data['analysis']['theoretical_bound'] is not None:
            names_with_bound.append(name)
            numerical_times.append(data['sim_hocbf'].recovery_time or T_max)
            theoretical_bounds.append(data['analysis']['theoretical_bound'])

    if names_with_bound:
        x_pos = np.arange(len(names_with_bound))
        width = 0.35
        ax.bar(x_pos - width/2, theoretical_bounds, width, label='Theoretical bound', color='C0', alpha=0.7)
        ax.bar(x_pos + width/2, numerical_times, width, label='Numerical', color='C1', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names_with_bound)
        ax.set_ylabel('Time [s]')
        ax.set_title('Recovery Time: Theory vs Numerical')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Experiment 1: r=2 Double Integrator (k={k})', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / "fig1_r2_recovery.png"
    plt.savefig(save_path, dpi=config['plot']['dpi'], bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_exp1()
