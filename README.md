# Rop05Pixe：固定半移＋PixelStitch＋论文适配模块

这是可独立训练和测试的项目。RopStitch 与 PixelStitch 所需源码已经放在 `vendor/`，运行时不引用服务器上的另外两个源码目录。数据集只需包含同名配对的 `input1/` 和 `input2/`。

固定四角分解系数为 0.5；RopStitch 预测单应与双向 TPS，PixelStitch 在 TPS 后预测残差光流。论文适配的新代码全部集中在 `paper_modules.py`，包括多尺度双注意相关适配器、低频结构/高频边缘监督，以及差异感知缝合掩码与边界锚定/平滑损失。原基线文件只保留少量调用钩子，便于关闭模块做消融。

## 从零训练

默认日程是 `geometry:100,residual:50,fusion:20`：

```bash
conda activate ropstitch
cd /workspace/python_pro/test1/sxl/Rop05Pixe

python ResearchRuntime/launch.py --gpu 1 --name paper_scratch_v1 --program stitch.py -- \
  train --data /path/to/training --out runs/paper_scratch_v1 --gpu 1
```

## 从原 150 轮基线继续训练新增模块

```bash
python ResearchRuntime/launch.py --gpu 1 --name paper_adapter_v1 --program stitch.py -- \
  train --data /path/to/training \
  --base-checkpoint checkpoints/fixed_half_pixel_epoch150.pt \
  --schedule geometry:10,residual:10,fusion:20 \
  --out runs/paper_adapter_v1 --gpu 1
```

## 测试和统一对照

```bash
python ResearchRuntime/launch.py --gpu 1 --name paper_test_v1 --program stitch.py -- \
  test --data /path/to/testing --checkpoint runs/paper_adapter_v1/last.pt \
  --out runs/paper_test_v1 --gpu 1

python ResearchRuntime/launch.py --gpu 1 --name paper_compare_v1 --program stitch.py -- \
  compare --data /path/to/original/1105-pair-testing \
  --checkpoint runs/paper_adapter_v1/last.pt --out runs/paper_compare_v1
```

普通测试会输出 `blend.png`、`seam_probability.png`、两张变形图和有效掩码。对照入口输出 mPSNR、mSSIM、覆盖率和形变指标。原基线的 25.128 dB/0.7930 不会被覆盖；新模块是否提升必须以相同 1,105 对全量结果为准。

每轮原子保存 `last.pt`。断点续训应使用新的输出目录并传入 `--resume`，训练清单、配置和已完成日程前缀必须保持一致。详细说明见 `STANDALONE.md`。
