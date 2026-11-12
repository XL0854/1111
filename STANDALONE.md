# 固定四角半移＋PixelStitch

训练和测试入口是 `stitch.py`。项目内置 RopStitch 几何模型、PixelStitch 光流源码及对照用权重。服务器已有的 `ropstitch` Conda 环境提供 PyTorch、CUDA、TorchVision、OpenCV、NumPy、SciPy 和 scikit-image 等依赖。

训练集和测试集各自包含 `input1/` 与 `input2/`，按文件名主干配对。将下面的数据路径替换为实际目录。示例使用物理 GPU1，运行前请确认该卡可用。

```bash
conda activate ropstitch
cd /workspace/python_pro/test1/sxl/FixedHalfPixelPipeline

python ResearchRuntime/launch.py --gpu 1 --name scratch_v1 --program stitch.py -- \
  train --data /path/to/training --out runs/scratch_v1 --gpu 1

python ResearchRuntime/launch.py --gpu 1 --name test_v1 --program stitch.py -- \
  test --data /path/to/testing --checkpoint runs/scratch_v1/last.pt \
  --out runs/test_v1 --gpu 1
```

默认训练从随机权重开始，先跑 100 轮几何，再跑 50 轮残差。用 `--schedule geometry:60,residual:20` 可以调整轮数。每轮保存 `last.pt`，可在新输出目录用 `--resume` 续训。

普通测试输出拼接图、两张变形图、有效掩码和光度指标。此前得到 mPSNR 25.128 dB、mSSIM 0.7930 的检查点已放在 `checkpoints/fixed_half_pixel_epoch150.pt`。要复算上述指标，使用原先的 1,105 对清单及同一比较入口：

```bash
python ResearchRuntime/launch.py --gpu 1 --name reproduce_v1 --program stitch.py -- \
  compare --data /path/to/original/1105-pair-testing \
  --checkpoint checkpoints/fixed_half_pixel_epoch150.pt --out runs/reproduce_v1
```

任务日志位于项目内的 `experiments/run_registry/<任务名>/output.log`。
