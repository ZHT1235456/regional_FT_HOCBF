"""
补充实验：对比不同控制器强度下的 HOCBF 效果。

说明：主实验已使用 u_nom=0，本实验展示控制器强度的影响。
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
from src.simulators import simulate_with_hocbf, simulate_without_constraint


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_supplementary():
    config = load_config()
    system = DoubleIntegrator(q_max=config['system']['q_max'])
    k = config['hocbf']['k']
    T_max = config['simulation']['T_max']
    dt = config['simulation']['dt']

    # ψ₁<0 的初始条件
    x0 = np.array([1.5, 1.0])  # h=-0.5, ψ₁=-1.5

    psi = compute_psi(system, x0, k)
    analysis = recovery_analysis(psi)
    print(f"初始条件: q0={x0[0]}, qdot0={x0[1]}")
    print(f"h(x0)={psi[0]:.2f}, psi_1={psi[1]:.2f}")
    print(f"推论条件: {analysis['sufficient_message']}")

    # 不同强度的名义控制器
    controllers = {
        'Strong PD (Kp=2, Kd=2)': lambda x, t: -2.0*(x[0]-0.5) - 2.0*x[1],
        'Weak PD (Kp=0.5, Kd=0.5)': lambda x, t: -0.5*(x[0]-0.5) - 0.5*x[1],
        'Very Weak (Kp=0.2)': lambda x, t: -0.2*(x[0]-0.5),
        'Zero (no control)': lambda x, t: 0.0,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, u_nom) in enumerate(controllers.items()):
        ax = axes[idx]

        # 有 HOCBF
        sim_hocbf = simulate_with_hocbf(system, x0, k, u_nom, T_max, dt)
        # 无 HOCBF
        sim_free = simulate_without_constraint(system, x0, u_nom, T_max, dt)

        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='h=0 (safe)')
        ax.plot(sim_hocbf.t, sim_hocbf.h, 'C0', label=f'HOCBF (recovered={sim_hocbf.recovered})')
        ax.plot(sim_free.t, sim_free.h, 'C1', linestyle=':', label=f'Free (recovered={sim_free.recovered})')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('h(t)')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T_max)

        print(f"\n{name}:")
        print(f"  HOCBF: recovered={sim_hocbf.recovered}, time={sim_hocbf.recovery_time}")
        print(f"  Free:  recovered={sim_free.recovered}, time={sim_free.recovery_time}")

    plt.suptitle(f'Supplementary: Why psi_1<0 can still recover\n'
                 f'Initial: q0={x0[0]}, qdot0={x0[1]}, psi_1={psi[1]:.2f}', fontsize=13)
    plt.tight_layout()

    save_dir = Path(__file__).parent.parent / "results"
    save_path = save_dir / "fig5_supplementary_psi1_negative.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_supplementary()
