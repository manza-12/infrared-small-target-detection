# 红外小目标检测/分割课程项目

## 项目简介

本仓库是红外小目标检测/分割课程大作业代码仓库。当前主要完成的是 ACM segmentation line 的 SIRST 测试集实验，围绕红外小目标分割任务进行评估、结果整理和可视化。

仓库提交版本只包含代码、文档、表格和少量代表性图片；不包含数据集、checkpoint、完整训练日志和大量预测图。

## 当前已完成内容

当前最终主表包含四个 ACM 配置：

* FPN + BiLocal
* FPN + AsymBi
* UNet + AsymBi
* UNet + BiGlobal

测试集为 `data/sirst/idx_427/test.txt`，共 640 张图像。当前 ACM 代码中 `mode='val'` 读取的就是该 `test.txt`，因此训练阶段 validation 指标与 final test 指标一致。

已整理内容包括：

* IoU、nIoU、Pd、Fa 指标
* 指标汇总表
* 指标柱状图
* 三栏可视化：Original / Ground Truth / Prediction
* bad case 候选
* 红外小目标 3D 灰度响应图
* 可直接放入论文/PPT 的报告片段

## 主要结果表

Fa 定义为 `false_alarm_pixels / total_images`。

| Backbone | Fusion | IoU | nIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|
| FPN | BiLocal | 0.2664489244 | 0.3093132385 | 0.9875 | 704.678125 |
| FPN | AsymBi | 0.4059420923 | 0.4855873167 | 0.990625 | 360.0703125 |
| UNet | AsymBi | 0.3262435852 | 0.3772964333 | 0.9953125 | 530.640625 |
| UNet | BiGlobal | 0.3883507577 | 0.4837368390 | 0.99375 | 389.1375 |

说明：FPN+AsymBi 的正式选用 checkpoint 是 best-IoU checkpoint。训练过程中 epoch 22 曾取得 best nIoU 0.49409，但当前 final test 表按 epoch 26 best-IoU checkpoint 汇总。UNet+AsymBi 使用最终确认的 `asymbi.pt`，UNet+BiGlobal 使用完整 result 目录中的 `.pkl` checkpoint，不混用 `.pt` 和 `.pkl` 指标。

## 目录结构说明

```text
.
├── acm/                         # ACM segmentation line 源码
├── tools/                       # 独立评估、Pd/Fa、可视化和交付包生成脚本
├── docs/                        # 训练记录、最终测试结果、仓库提交说明
├── final_experiment_pack/        # 课程报告可用的表格、图和报告片段
│   ├── README.md
│   ├── tables/
│   ├── figures/
│   ├── report_snippets/
│   ├── selected_examples/        # 少量代表性图片，可提交
│   └── FAILED_ITEMS.md
├── run.py                       # 分工训练入口，不建议在提交检查时运行
├── setup_data.py                # 数据准备脚本
├── infer.py                     # YOLO 方向推理脚本
├── requirements.txt
└── .gitignore
```

以下目录或文件通常不进入 Git：

* `data/`
* `result/`
* `runs/`
* `final_test_outputs/`
* checkpoint 权重文件，如 `.pkl`、`.pt`、`.pth`
* 完整训练日志和 TensorBoard event 文件
* 大量预测图目录

## 环境说明

本次 person1 实验使用环境：

* Python 3.11
* torch 2.6.0 + cu124
* CUDA available
* GPU: NVIDIA GeForce RTX 4060 Laptop GPU 8GB

最小 Python 依赖见 `requirements.txt`。实际复现实验时需要根据本机 CUDA 和 PyTorch 版本安装兼容的 `torch` / `torchvision`。

## 如何运行评估脚本

checkpoint 文件不包含在仓库中，需要自行放置到 `result/` 目录下，并保持下列相对路径，或在命令中替换为自己的路径。

FPN + BiLocal:

```powershell
python tools/evaluate_acm_checkpoint.py `
  --backbone-mode FPN `
  --fuse-mode BiLocal `
  --checkpoint "result/2026-07-10-01-38-03_FPN_BiLocal/checkpoint/Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir final_test_outputs/person1/FPN_BiLocal
```

FPN + AsymBi:

```powershell
python tools/evaluate_acm_checkpoint.py `
  --backbone-mode FPN `
  --fuse-mode AsymBi `
  --checkpoint "result/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir final_test_outputs/person1/FPN_AsymBi
```

Pd/Fa 评估示例：

```powershell
python tools/evaluate_acm_pd_fa.py `
  --backbone-mode FPN `
  --fuse-mode AsymBi `
  --checkpoint "result/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir final_test_outputs/person1/FPN_AsymBi
```

生成三栏预测图示例：

```powershell
python tools/visualize_acm_predictions.py `
  --backbone-mode FPN `
  --fuse-mode AsymBi `
  --checkpoint "result/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl" `
  --data-root data/sirst `
  --split-file idx_427/test.txt `
  --output-dir final_experiment_pack/visualizations/FPN_AsymBi `
  --max-images 50
```

## 输出文件说明

主要可交付材料位于：

* `final_experiment_pack/tables/final_metrics_all_available.csv`
* `final_experiment_pack/tables/final_metrics_person1.csv`
* `final_experiment_pack/tables/unavailable_experiments_note.csv`
* `final_experiment_pack/figures/`
* `final_experiment_pack/report_snippets/`
* `final_experiment_pack/selected_examples/`
* `docs/person1_final_test_results.csv`
* `docs/person1_final_test_report.md`
* `docs/acm_results_table.md`

## 实验限制

* 当前仓库不包含数据集和 checkpoint。
* person2/person3/person4 尚未全部纳入最终汇总。
* PConv k=3/4、SD Loss δ 消融、SIRST→NUAA 泛化、YOLO mAP50 暂未完成或不在当前 ACM 分割线中。
* ACM 分割指标 IoU/nIoU/Pd/Fa 与 YOLO 检测框 mAP50 不应混在同一指标表中直接比较。
