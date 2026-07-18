# 仓库提交说明

## 本仓库提交内容

本仓库用于课程大作业代码提交，建议提交以下内容：

* ACM segmentation line 相关代码
* 独立评估和可视化脚本：`tools/`
* 实验说明与结果文档：`docs/`
* 最小依赖文件：`requirements.txt`
* 仓库说明：`README.md`
* 结果表格、指标图、报告片段和少量代表性图片：`deliverables/final_experiment_pack/`

## 未提交内容及原因

以下内容通过 `.gitignore` 排除，不建议提交到 GitHub：

* `data/`：数据集体积较大，且可能涉及课程组内数据分发约定。
* `experiments/training_runs/`、`checkpoints/`、`*.pkl`、`*.pt`、`*.pth`：checkpoint 和模型权重文件较大。
* `person1_full_train_log.txt`、TensorBoard event 文件和完整日志：训练日志较大，提交价值有限。
* `deliverables/final_experiment_pack/visualizations/`、`bad_cases/`、`gray_surface/`：包含大量生成图片，只保留少量样例到 `selected_examples/`。

## 如何复现实验

1. 准备 Python 环境，并安装 `requirements.txt` 中的最小依赖。
2. 将 SIRST 数据按项目要求放置到 `data/sirst/`，包含 `images/`、`masks/` 和 `idx_427/test.txt`。
3. 将 person1 checkpoint 放置到 README 中给出的 `experiments/training_runs/` 路径，或在命令中替换为本机 checkpoint 路径。
4. 运行 `tools/evaluate_acm_checkpoint.py` 得到 IoU/nIoU。
5. 运行 `tools/evaluate_acm_pd_fa.py` 得到 Pd/Fa。
6. 如需可视化，运行 `tools/visualize_acm_predictions.py`。

## 如何用于论文/PPT

* 量化结果表：使用 `deliverables/final_experiment_pack/tables/final_metrics_person1.csv`。
* 指标柱状图：使用 `deliverables/final_experiment_pack/figures/`。
* 代表性可视化：使用 `deliverables/final_experiment_pack/selected_examples/`。
* 方法与实验说明：参考 `deliverables/final_experiment_pack/report_snippets/experiment_setup.md`。
* 结果分析：参考 `deliverables/final_experiment_pack/report_snippets/results_analysis_person1.md`。
* 局限性说明：参考 `deliverables/final_experiment_pack/report_snippets/limitations.md`。
* 指标定义：参考 `deliverables/final_experiment_pack/report_snippets/metric_definitions.md`。

## person1 当前结果是否完整

person1 的 FPN+BiLocal 和 FPN+AsymBi 已完成正式训练、final test、IoU/nIoU/Pd/Fa 汇总、指标图、三栏图、bad case 候选和 3D 灰度响应图整理。

当前最终汇总结果：

| Backbone | Fusion | IoU | nIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|
| FPN | BiLocal | 0.2664489244 | 0.3093132385 | 0.9875 | 704.678125 |
| FPN | AsymBi | 0.4059420923 | 0.4855873167 | 0.990625 | 360.0703125 |

Fa 定义为 `false_alarm_pixels / total_images`。
