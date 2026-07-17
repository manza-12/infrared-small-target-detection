# Final Experiment Pack

This folder collects the final deliverable materials for the ACM segmentation line on the SIRST test split. It contains tables, figures, report snippets, and a small number of representative images. It does not contain datasets or checkpoints.

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

Full generated qualitative results are stored in `visualizations/` and bad-case candidates are stored in `bad_cases/`. These folders may contain many generated images and are normally excluded from GitHub by `.gitignore`.

The metadata table is:

- `bad_cases/bad_case_table.csv`

## Gray Surface

The 3D gray response examples are stored in `gray_surface/`. These images describe the SIRST infrared small-target data itself and are not model-dependent, so they are reused for person2.

## Selected Examples

Small representative images for GitHub or PPT preview are stored in `selected_examples/`. The selected set is intentionally limited to keep the repository lightweight.

## Report Snippets

Copy-ready report fragments are stored in `report_snippets/`:

- `experiment_setup.md`
- `results_analysis_all_available.md`
- `limitations.md`
- `metric_definitions.md`

## Notes

The final SIRST test split is `data/sirst/idx_427/test.txt` with 640 images and 0 train/test overlap. In the current ACM code, validation mode reads this same split, so this should be described clearly in reports.

YOLO mAP50, PConv ablations, SD Loss ablations, and NUAA cross-dataset testing are not part of the current ACM six-model segmentation main table.
