# Regional Finite-Time Recovery for HOCBF

仿真实验验证高阶控制屏障函数（HOCBF）的区域有限时间恢复性质。

## 理论背景

对于相对度 $r \ge 2$ 的安全约束，通过定义辅助变量 $\psi_i = (d/dt + k)^i h$，高阶屏障约束 $\psi_r \ge 0$ 可嵌入 QP 安全滤波器。

主要结论：
- **定理 1**：若恢复多项式 $P_{r-1}(t) = \sum_{i=0}^{r-1} \frac{\psi_i(x_0)}{i!} t^i$ 存在正根，则系统在有限时间内恢复
- **推论 1**：简单充分条件 $\psi_1(x_0) > 0, \psi_i(x_0) \ge 0$（$i \ge 2$），恢复时间界 $T \le -h(x_0)/\psi_1(x_0)$

## 项目结构

```
├── src/
│   ├── systems.py        # 系统动力学（双/三阶积分器）
│   ├── hocbf.py          # HOCBF 辅助变量计算
│   ├── recovery.py       # 恢复多项式与理论时间界
│   └── simulators.py     # QP 安全滤波器仿真器
├── experiments/
│   ├── config.yaml       # 实验参数配置
│   ├── run_exp1_r2_recovery.py
│   ├── run_exp2_r3_recovery.py
│   ├── run_exp3_comparison.py
│   └── run_exp4_region.py
├── tests/
│   └── test_hocbf.py
└── results/              # 自动生成的图表
```

## 实验内容

1. **实验 1**：r=2 双积分器有限时间恢复验证
2. **实验 2**：r=3 三阶积分器有限时间恢复验证
3. **实验 3**：不同 k 值对恢复时间的影响
4. **实验 4**：恢复区域可视化

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行单个实验
python experiments/run_exp1_r2_recovery.py

# 运行全部实验
python -m experiments.run_all

# 运行测试
pytest tests/
```

## 参考文献

- Ames et al., "Control barrier function based quadratic programs," CDC 2014
- Nguyen & Sreenath, "Exponential control barrier functions," ACC 2016
- Xiao & Belta, "Control barrier functions for systems with high relative degree," CDC 2019
