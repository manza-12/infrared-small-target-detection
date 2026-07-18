# Final Experiment Pack Index

## Core Tables

- Six-model main table: `tables/final_metrics_all_available.csv`
- Person1 table: `tables/final_metrics_person1.csv`
- Backbone/fusion ablation table: `tables/backbone_fusion_ablation.csv`
- Unavailable experiment note: `tables/unavailable_experiments_note.csv`

## Metric Figures

- IoU comparison: `figures/all_available_iou_comparison.png`
- nIoU comparison: `figures/all_available_niou_comparison.png`
- Pd comparison: `figures/all_available_pd_comparison.png`
- Fa comparison: `figures/all_available_fa_comparison.png`
- FPN fusion comparison: `figures/fusion_comparison_fpn.png`
- UNet fusion comparison: `figures/fusion_comparison_unet.png`
- Person1 IoU comparison: `figures/person1_iou_comparison.png`
- Person1 nIoU comparison: `figures/person1_niou_comparison.png`

## Report Snippets

- Experiment setup: `report_snippets/experiment_setup.md`
- All-available results analysis: `report_snippets/results_analysis_all_available.md`
- Person1 results analysis: `report_snippets/results_analysis_person1.md`
- Metric definitions: `report_snippets/metric_definitions.md`
- Limitations: `report_snippets/limitations.md`

## Qualitative Materials

- Selected examples: `selected_examples/`
- Full visualizations: `visualizations/`
- Bad cases: `bad_cases/`
- Bad-case table: `bad_cases/bad_case_table.csv`
- 3D gray response plots: `gray_surface/`

## Related Experiment Outputs

- Final test outputs: `../../experiments/final_test_outputs/`
- Training checkpoints and TensorBoard event files: `../../experiments/training_runs/`
- Background training logs: `../../experiments/background_runs/`

## Notes

This package was moved from the former root-level `final_experiment_pack/` path to `deliverables/final_experiment_pack/`. The files were moved with `git mv`; no experiment artifact content was intentionally changed by the directory reorganization.