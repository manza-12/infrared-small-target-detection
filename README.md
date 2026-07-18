# 红外小目标检测/分割课程项目

## 项目简介

本仓库是红外小目标检测/分割课程大作业代码仓库。当前整理完成的是 ACM segmentation line 在 SIRST 固定测试划分上的实验、评估与可视化材料。

GitHub 提交版本只包含代码、文档、表格、图表和少量代表性图片；不包含数据集、checkpoint、完整训练日志、完整预测图和 `final_test_outputs/`。

## 当前已完成内容

当前 ACM 主表已经完成完整的 2 backbone x 3 fusion strategy 矩阵，共六个模型：

- FPN + BiLocal
- FPN + AsymBi
- FPN + BiGlobal
- UNet + BiLocal
- UNet + AsymBi
- UNet + BiGlobal

统一测试集为 `data/sirst/idx_427/test.txt`，共 640 张图像。当前 ACM 代码中 validation 使用的也是该 test split，因此 final test 与训练期间 validation 指标一致或非常接近是合理的，但这不是外部独立泛化测试。

已整理内容包括：

- IoU、nIoU、Pd、Fa 指标
- 六模型主表
- backbone/fusion 消融表
- 指标柱状图
- 三栏可视化图：Original / Ground Truth / Prediction
- bad case 候选图
- SIRST 目标 3D 灰度响应图
- 可直接放入论文或 PPT 的报告片段

## 主要结果表

Fa 定义为 `false_alarm_pixels / total_images`。

| Person | Backbone | Fusion | IoU | nIoU | Pd | Fa |
|---|---|---|---:|---:|---:|---:|
| person1 | FPN | BiLocal | 0.2664489244 | 0.3093132385 | 0.9875000 | 704.678125 |
| person1 | FPN | AsymBi | 0.4059420923 | 0.4855873167 | 0.9906250 | 360.070312 |
| person2 | FPN | BiGlobal | 0.3660601796 | 0.4457140235 | 0.9890625 | 440.821875 |
| person2 | UNet | BiLocal | 0.3772599337 | 0.4348144315 | 0.9859375 | 415.790625 |
| person3 | UNet | AsymBi | 0.3262435852 | 0.3772964333 | 0.9953125 | 530.640625 |
| person4 | UNet | BiGlobal | 0.3883507577 | 0.4837368390 | 0.9937500 | 389.137500 |

## 目录结构说明

```text
.
|-- acm/                         # ACM segmentation line 源码
|-- tools/                       # 评估、Pd/Fa、可视化和实验包整理脚本
|-- docs/                        # 训练记录、测试记录和仓库提交说明
|-- final_experiment_pack/        # 课程报告可用的表格、图和片段
|   |-- README.md
|   |-- tables/
|   |-- figures/
|   |-- report_snippets/
|   |-- selected_examples/
|   `-- FAILED_ITEMS.md
|-- run.py                       # 分工训练入口，提交整理时不要运行
|-- setup_data.py
|-- infer.py
|-- requirements.txt
`-- .gitignore
```

以下目录或文件通常不进入 Git：

- `data/`
- `result/`
- `runs/`
- `final_test_outputs/`
- `background_runs/`
- checkpoint 权重文件，如 `.pkl`、`.pt`、`.pth`
- 完整训练日志和 TensorBoard event 文件
- 大量预测图目录

## 环境说明

本地实验环境：

- Python 3.11
- torch 2.6.0 + cu124
- CUDA available
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU 8GB

最小 Python 依赖见 `requirements.txt`。实际复现实验时，需要根据本机 CUDA 版本安装兼容的 `torch` 和 `torchvision`。

## 如何运行评估脚本

checkpoint 文件不包含在仓库中，需要自行放置到 `result/` 目录，或在命令中替换为自己的路径。

FPN + BiGlobal 示例：

```powershell
python tools/evaluate_acm_checkpoint.py `
  --backbone-mode FPN `
  --fuse-mode BiGlobal `
  --checkpoint "result/2026-07-10-11-40-36_FPN_BiGlobal/checkpoint/Epoch- 29_IoU-0.3661_nIoU-0.4457.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir final_test_outputs/person2/FPN_BiGlobal_selected
```

UNet + BiLocal 的 Pd/Fa 示例：

```powershell
python tools/evaluate_acm_pd_fa.py `
  --backbone-mode UNet `
  --fuse-mode BiLocal `
  --checkpoint "result/2026-07-10-12-58-28_UNet_BiLocal/checkpoint/Epoch- 30_IoU-0.3773_nIoU-0.4348.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir final_test_outputs/person2/UNet_BiLocal_selected
```

生成三栏可视化图示例：

```powershell
python tools/visualize_acm_predictions.py `
  --backbone-mode FPN `
  --fuse-mode BiGlobal `
  --checkpoint "result/2026-07-10-11-40-36_FPN_BiGlobal/checkpoint/Epoch- 29_IoU-0.3661_nIoU-0.4457.pkl" `
  --data-root data/sirst `
  --split-file idx_427/test.txt `
  --output-dir final_experiment_pack/visualizations/FPN_BiGlobal `
  --max-images 50
```

## 输出文件说明

主要可交付材料位于：

- `final_experiment_pack/tables/final_metrics_all_available.csv`
- `final_experiment_pack/tables/backbone_fusion_ablation.csv`
- `final_experiment_pack/figures/`
- `final_experiment_pack/report_snippets/`
- `final_experiment_pack/selected_examples/`

## 实验限制

- 当前仓库不包含数据集和 checkpoint。
- 当前完成的是 ACM 分割线的 backbone 与 fusion strategy 消融。
- PConv k=3/4、SD Loss、SIRST 到 NUAA 泛化、YOLO mAP50 暂未纳入当前 ACM 六模型主表。
- ACM 分割指标 IoU/nIoU/Pd/Fa 与 YOLO 检测框 mAP50 不应混表比较。
