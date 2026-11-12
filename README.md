# 固定四角半位移 + PixelStitch 残差

训练和测试都固定四个角点的分解系数为 0.5。这里的中间平面由角点位移插值得到，与 PixelStitch 的矩阵平方根是两种构造。RopStitch 负责预测相对单应和双向 TPS，PixelStitch 的双向残差光流在 TPS 之后细化对齐。

默认从随机权重开始，几何阶段 100 轮，像素残差阶段 50 轮。几何阶段更新两个匹配分支、全局单应和 TPS；残差阶段冻结已训练的几何，更新像素光流。固定平面模式没有可训练的平面系数网络。

```bash
conda activate ropstitch
ROP=/workspace/python_pro/test1/RopStitch
PIXEL=/workspace/python_pro/test1/PixelStitch
cd "$ROP"
python ResearchRuntime/launch.py --gpu 1 --name fixed_half_pixel_scratch_v1 \
  --program FixedHalfPixelPipeline/full.py -- train \
  --rop "$ROP" --pixel "$PIXEL" --data "$ROP/data/UDIS-D/training" \
  --init scratch --plane-mode fixed-half --schedule geometry:100,residual:50 \
  --out "$ROP/FixedHalfPixelPipeline/runs/train_v1" --gpu 1
```

测试时自动从权重恢复固定平面模式和画布选项。传入已核对数据来源的测试清单，可逐对输出变形图、最终掩码、拼接图和指标：

```bash
python ResearchRuntime/launch.py --gpu 1 --name fixed_half_pixel_test_v1 \
  --program FixedHalfPixelPipeline/full.py -- test \
  --rop "$ROP" --pixel "$PIXEL" --data "$TEST_ROOT" --list "$TEST_LIST" \
  --checkpoint "$ROP/FixedHalfPixelPipeline/runs/train_v1/last.pt" \
  --out "$ROP/FixedHalfPixelPipeline/runs/test_v1" --gpu 1
```

每轮原子保存 `last.pt`。断点续训在新输出目录里使用 `--resume`，保持训练清单、配置和已完成的阶段前缀一致。
