"""
实验 3：不同 k 值对恢复时间的影响。
实验 4：恢复区域可视化。
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
from src.recovery import recovery_analysis
from src.simulators import simulate_with_hocbf


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_exp3_k_comparison():
    """实验 3：k 值对比"""
    config = load_config()
    system = DoubleIntegrator(q_max=config['system']['q_max'])
    T_max = config['simulation']['T_max']
    dt = config['simulation']['dt']
    x0 = np.array(config['exp3']['initial_condition'])
    k_list = config['exp3']['k_list']

    def u_nominal(x, t):
        q_ref = 0.5
        return -2.0 * (x[0] - q_ref) - 2.0 * x[1]

    print("=" * 60)
    print("实验 3：不同 k 值对恢复时间的影响")
    print("=" * 60)
    print(f"初始条件: q0={x0[0]}, q_dot0={x0[1]}")

    k_results = {}
    for k in k_list:
        psi = compute_psi(system, x0, k)
        analysis = recovery_analysis(psi)
        sim = simulate_with_hocbf(system, x0, k, u_nominal, T_max, dt)

        print(f"\n  k={k:.1f}: ψ1={psi[1]:.4f}, 理论界={analysis['theoretical_bound']}, "
              f"实际恢复={sim.recovery_time}")

        k_results[k] = {
            'psi': psi,
            'analysis': analysis,
            'sim': sim,
        }

    plot_exp3_results(k_results, x0, config)
    return k_results


def run_exp4_recovery_region():
    """实验 4：恢复区域可视化"""
    config = load_config()
    system = DoubleIntegrator(q_max=config['system']['q_max'])
    k = config['hocbf']['k']

    # 在 (q0, q_dot0) 平面上扫描
    q0_range = np.linspace(0.5, 2.5, 50)
    qdot0_range = np.linspace(-2.0, 2.0, 50)
    Q0, QDOT0 = np.meshgrid(q0_range, qdot0_range)

    # 计算理论恢复区域
    recovery_region = np.zeros_like(Q0)
    psi1_region = np.zeros_like(Q0)

    for i in range(Q0.shape[0]):
        for j in range(Q0.shape[1]):
            x0 = np.array([Q0[i, j], QDOT0[i, j]])
            psi = compute_psi(system, x0, k)
            psi1_region[i, j] = psi[1]

            # 检查推论条件
            if psi[0] < 0 and psi[1] > 0:
                recovery_region[i, j] = 1.0  # 满足推论条件

    # 计算恢复时间界
    time_bound = np.full_like(Q0, np.nan)
    for i in range(Q0.shape[0]):
        for j in range(Q0.shape[1]):
            if recovery_region[i, j] == 1.0:
                x0 = np.array([Q0[i, j], QDOT0[i, j]])
                psi = compute_psi(system, x0, k)
                time_bound[i, j] = -psi[0] / psi[1]

    plot_exp4_results(Q0, QDOT0, recovery_region, time_bound, psi1_region, k, config)


def plot_exp3_results(k_results, x0, config):
    """绘制实验 3 结果"""
    project_root = Path(__file__).parent.parent
    save_dir = project_root / config['plot']['save_dir']
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图 1: 不同 k 下的 h(t)
    ax = axes[0]
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='h=0')
    for k, data in k_results.items():
        ax.plot(data['sim'].t, data['sim'].h, label=f'k={k}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('h(t)')
    ax.set_title(f'Safety Function for Different k (q0={x0[0]}, qdot0={x0[1]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)

    # 图 2: 恢复时间 vs k
    ax = axes[1]
    ks = sorted(k_results.keys())
    numerical = [k_results[k]['sim'].recovery_time or config['simulation']['T_max'] for k in ks]
    theoretical = [k_results[k]['analysis']['theoretical_bound'] or 0 for k in ks]

    ax.plot(ks, numerical, 'o-', label='Numerical', color='C0')
    ax.plot(ks, theoretical, 's--', label='Theoretical bound', color='C1')
    ax.set_xlabel('k')
    ax.set_ylabel('Recovery Time [s]')
    ax.set_title('Recovery Time vs HOCBF Gain k')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 3: Effect of HOCBF Gain k', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / "fig3_k_comparison.png"
    plt.savefig(save_path, dpi=config['plot']['dpi'], bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()


def plot_exp4_results(Q0, QDOT0, recovery_region, time_bound, psi1_region, k, config):
    """绘制实验 4 结果"""
    project_root = Path(__file__).parent.parent
    save_dir = project_root / config['plot']['save_dir']
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    q_max = config['system']['q_max']

    # 图 1: 恢复区域
    ax = axes[0]
    im = ax.contourf(Q0, QDOT0, recovery_region, levels=[-0.5, 0.5, 1.5],
                     colors=['#ff9999', '#99ff99'], alpha=0.7)
    ax.contour(Q0, QDOT0, recovery_region, levels=[0.5], colors='green', linewidths=2)
    ax.axvline(x=q_max, color='r', linestyle='--', label=f'Safety boundary q={q_max}')
    ax.set_xlabel('q0')
    ax.set_ylabel('q_dot0')
    ax.set_title(f'Recovery Region (green = sufficient condition)\nk={k}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 2: 理论恢复时间界
    ax = axes[1]
    masked_time = np.ma.masked_invalid(time_bound)
    if not np.all(np.isnan(time_bound)):
        c = ax.contourf(Q0, QDOT0, masked_time, levels=15, cmap='viridis')
        plt.colorbar(c, ax=ax, label='Recovery time bound [s]')
    ax.axvline(x=q_max, color='r', linestyle='--', label=f'Safety boundary')
    ax.set_xlabel('q0')
    ax.set_ylabel('q_dot0')
    ax.set_title(f'Theoretical Recovery Time Bound\nk={k}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 4: Recovery Region Visualization', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / "fig4_recovery_region.png"
    plt.savefig(save_path, dpi=config['plot']['dpi'], bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_exp3_k_comparison()
    run_exp4_recovery_region()
