"""
实验 2：r=3 三阶积分器有限时间恢复验证。

验证目标：
1. 满足推论条件 (h<0, ψ1>0, ψ2>=0) 的初始状态能在理论时间界内恢复
2. 仅 ψ1>0 但 ψ2<0 时，可能需要多项式条件（展示反例现象）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from src.systems import TripleIntegrator
from src.hocbf import compute_psi
from src.recovery import recovery_analysis
from src.simulators import simulate_with_hocbf


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_exp2():
    """运行实验 2"""
    config = load_config()
    system = TripleIntegrator(q_max=config['system']['q_max'])
    k = config['hocbf']['k']
    T_max = config['simulation']['T_max']
    dt = config['simulation']['dt']

    # 名义控制器设为 0，纯粹验证 HOCBF 的恢复能力
    def u_nominal(x, t):
        return 0.0

    ics = config['exp2']['initial_conditions']
    results = {}

    print("=" * 60)
    print("实验 2：r=3 三阶积分器有限时间恢复验证")
    print("=" * 60)

    for name, x0_list in ics.items():
        x0 = np.array(x0_list)
        q0, qdot0, qddot0 = x0

        psi = compute_psi(system, x0, k)
        analysis = recovery_analysis(psi)

        print(f"\n--- 初始条件 {name}: q0={q0}, q_dot0={qdot0}, q_ddot0={qddot0} ---")
        print(f"  h(x0) = {analysis['h0']:.4f}")
        print(f"  ψ0 = {psi[0]:.4f}, ψ1 = {psi[1]:.4f}, ψ2 = {psi[2]:.4f}")
        print(f"  推论条件: {analysis['sufficient_message']}")
        if analysis['theoretical_bound'] is not None:
            print(f"  理论恢复时间界: T <= {analysis['theoretical_bound']:.4f}")
        if analysis['polynomial_root'] is not None:
            print(f"  多项式正根: T = {analysis['polynomial_root']:.4f}")

        sim = simulate_with_hocbf(system, x0, k, u_nominal, T_max, dt)

        print(f"  [HOCBF] 恢复: {sim.recovered}, 恢复时间: {sim.recovery_time}")

        if analysis['theoretical_bound'] is not None and sim.recovered:
            ratio = sim.recovery_time / analysis['theoretical_bound']
            print(f"  数值/理论 比值: {ratio:.4f}")

        results[name] = {
            'x0': x0, 'psi': psi, 'analysis': analysis, 'sim': sim,
        }

    plot_exp2_results(results, k, config)
    return results


def plot_exp2_results(results, k, config):
    """绘制实验 2 结果"""
    project_root = Path(__file__).parent.parent
    save_dir = project_root / config['plot']['save_dir']
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'A1': 'C0', 'A2': 'C1', 'C1': 'C2'}
    q_max = config['system']['q_max']

    # 图 1: q(t)
    ax = axes[0]
    ax.axhline(y=q_max, color='r', linestyle='--', alpha=0.7, label='Safety boundary')
    for name, data in results.items():
        ax.plot(data['sim'].t, data['sim'].x[:, 0], color=colors.get(name, 'gray'),
                label=f"{name}: q0={data['x0'][0]}")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('q(t)')
    ax.set_title('Position Trajectories')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图 2: h(t)
    ax = axes[1]
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    for name, data in results.items():
        ax.plot(data['sim'].t, data['sim'].h, color=colors.get(name, 'gray'),
                label=f"{name}: psi1={data['psi'][1]:.2f}, psi2={data['psi'][2]:.2f}")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('h(t)')
    ax.set_title('Safety Function h(t)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图 3: ψ_r(t)
    ax = axes[2]
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    for name, data in results.items():
        ax.plot(data['sim'].t, data['sim'].psi_r, color=colors.get(name, 'gray'),
                label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$\psi_r(t)$')
    ax.set_title(r'HOCBF Constraint $\psi_r$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Experiment 2: r=3 Triple Integrator (k={k}, u_nom=0)', fontsize=14)
    plt.tight_layout()

    save_path = save_dir / "fig2_r3_recovery.png"
    plt.savefig(save_path, dpi=config['plot']['dpi'], bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_exp2()
