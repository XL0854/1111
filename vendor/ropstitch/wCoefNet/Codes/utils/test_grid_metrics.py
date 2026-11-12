#!/usr/bin/env python3
import numpy as np
import sys
import os

# add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grid_metrics import compute_grid_metrics


def create_regular_grid(H_cells=12, W_cells=12, width=512.0, height=512.0):
    x = np.linspace(0.0, width, W_cells + 1)
    y = np.linspace(0.0, height, H_cells + 1)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1)


def run_unit_tests():
    print("================== Running Grid Metrics Unit Tests ==================")
    grid_orig = create_regular_grid(12, 12, 512.0, 512.0)

    # 1. 规则网格测试
    m_reg = compute_grid_metrics(grid_orig, grid_orig)
    print("1. Regular Grid Metrics:", m_reg)
    assert m_reg["folding_ratio"] == 0.0, f"Regular grid folding_ratio should be 0, got {m_reg['folding_ratio']}"
    assert m_reg["degenerate_ratio"] == 0.0, f"Regular grid degenerate_ratio should be 0, got {m_reg['degenerate_ratio']}"
    assert abs(m_reg["bending_degree"]) < 1e-6, f"Regular grid bending should be ~0, got {m_reg['bending_degree']}"
    assert abs(m_reg["avg_stretch"]) < 1e-6, f"Regular grid avg_stretch should be ~0, got {m_reg['avg_stretch']}"
    assert abs(m_reg["scale_variation"]) < 1e-6, f"Regular grid scale_variation should be ~0, got {m_reg['scale_variation']}"
    print("   [PASS] 规则网格: 翻折率=0, 拉伸=0, 弯曲=0, 尺度变化=0\n")

    # 2. 平移网格测试 (x+100, y-50)
    grid_trans = grid_orig.copy()
    grid_trans[..., 0] += 100.0
    grid_trans[..., 1] -= 50.0
    m_trans = compute_grid_metrics(grid_trans, grid_orig)
    print("2. Translated Grid Metrics:", m_trans)
    assert m_trans["folding_ratio"] == 0.0, "Translated grid should have 0 folding"
    assert abs(m_trans["avg_stretch"]) < 1e-5, f"Translated grid avg_stretch should be ~0, got {m_trans['avg_stretch']}"
    assert abs(m_trans["bending_degree"]) < 1e-5, f"Translated grid bending should be ~0, got {m_trans['bending_degree']}"
    print("   [PASS] 平移网格: 拉伸接近0, 弯曲接近0, 翻折率=0\n")

    # 3. 均匀缩放网格测试 (scale = 2.5)
    grid_scale = grid_orig * 2.5
    m_scale = compute_grid_metrics(grid_scale, grid_orig)
    print("3. Uniformly Scaled Grid Metrics:", m_scale)
    assert m_scale["folding_ratio"] == 0.0, "Scaled grid should have 0 folding"
    assert abs(m_scale["scale_variation"]) < 1e-5, f"Scaled grid scale_variation should be ~0 (uniform), got {m_scale['scale_variation']}"
    assert abs(m_scale["avg_stretch"]) < 1e-5, f"Uniformly scaled grid avg_stretch should be ~0 (isotropic), got {m_scale['avg_stretch']}"
    assert abs(m_scale["bending_degree"]) < 1e-5, f"Scaled grid bending should be ~0, got {m_scale['bending_degree']}"
    print("   [PASS] 均匀缩放网格: 尺度变化方差接近0 (完全可预测), 翻折率=0\n")

    # 4. 手工翻折网格测试: 将中心一个内部顶点剧烈拉向反向，使相邻三角形翻转
    grid_fold = grid_orig.copy()
    # 交换点 (6, 6) 和 (5, 6) 或者将点 (6,6) 移动到点 (3, 3)
    grid_fold[6, 6] = grid_orig[4, 4]
    m_fold = compute_grid_metrics(grid_fold, grid_orig)
    print("4. Folded Grid Metrics:", m_fold)
    assert m_fold["folding_ratio"] > 0.0, f"Folded grid must have folding_ratio > 0, got {m_fold['folding_ratio']}"
    print(f"   [PASS] 手工翻折网格: 翻折率={m_fold['folding_ratio']:.4f} > 0\n")

    # 5. 手工弯曲网格测试: 给网格添加正弦波弯曲波形 y += 50 * sin(x)
    grid_bent = grid_orig.copy()
    grid_bent[..., 1] += 40.0 * np.sin(grid_orig[..., 0] / 50.0)
    m_bent = compute_grid_metrics(grid_bent, grid_orig)
    print("5. Bent Grid Metrics:", m_bent)
    assert m_bent["bending_degree"] > m_reg["bending_degree"] + 0.1, (
        f"Bent grid bending degree ({m_bent['bending_degree']}) must significantly exceed regular ({m_reg['bending_degree']})"
    )
    print(f"   [PASS] 手工弯曲网格: 弯曲指标显著增加 (从 {m_reg['bending_degree']:.6f} 升至 {m_bent['bending_degree']:.6f})\n")

    print("================== ALL UNIT TESTS PASSED SUCCESSFULLY! ==================")


if __name__ == "__main__":
    run_unit_tests()
