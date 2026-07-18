# Final Experiment Pack

This folder collects the final deliverable materials for the ACM segmentation line on the SIRST test split. It contains tables, figures, report snippets, representative images, full prediction visualizations, bad cases, gray-surface plots, and packaging logs.

This folder does not contain the dataset. Related checkpoints are stored under `../../experiments/training_runs/`, and related final test outputs are stored under `../../experiments/final_test_outputs/`.

## Included Models

The final main table contains six models, forming a complete 2 backbones x 3 fusion strategies matrix:

| Person | Backbone | Fusion | Selected weight |
|---|---|---|---|
| person1 | FPN | BiLocal | `.pkl` |
| person1 | FPN | AsymBi | `.pkl` |
| person2 | FPN | BiGlobal | `.pkl` |
| person2 | UNet | BiLocal | `.pkl` |
| person3 | UNet | AsymBi | `asymbi.pt` |
| person4 | UNet | BiGlobal | `.pkl` |

## Main Table

Main quantitative table:

- `tables/final_metrics_all_available.csv`

Backbone and fusion ablation table:

- `tables/backbone_fusion_ablation.csv`

## Figures

Metric figures are stored in `figures/`:

- `all_available_iou_comparison.png`
- `all_available_niou_comparison.png`
- `all_available_pd_comparison.png`
- `all_available_fa_comparison.png`
- `fusion_comparison_fpn.png`
- `fusion_comparison_unet.png`

## Visualizations and Bad Cases

Full generated qualitative results are stored in `visualizations/`, and bad-case candidates are stored in `bad_cases/`. The metadata table is `bad_cases/bad_case_table.csv`.

## Gray Surface

The 3D gray response examples are stored in `gray_surface/`. These images describe the SIRST infrared small-target data itself and are not model-dependent.

## Selected Examples

Representative images for GitHub or PPT preview are stored in `selected_examples/`.

## Report Snippets

Copy-ready report fragments are stored in `report_snippets/`:

- `experiment_setup.md`
- `results_analysis_all_available.md`
- `limitations.md`
- `metric_definitions.md`

## Navigation

Use `INDEX.md` for a compact list of important tables, figures, snippets, checkpoints, and final test outputs.

## Notes

The final SIRST test split is `data/sirst/idx_427/test.txt` with 640 images and 0 train/test overlap. In the current ACM code, validation mode reads this same split, so this should be described clearly in reports.

YOLO mAP50, PConv ablations, SD Loss ablations, and NUAA cross-dataset testing are not part of the current ACM six-model segmentation main table.