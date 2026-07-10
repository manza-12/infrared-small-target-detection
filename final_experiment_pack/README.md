# Final Experiment Pack

This folder collects the final deliverable materials for the ACM segmentation line on the SIRST test split. It contains tables, figures, report snippets, and a small number of representative images. It does not contain datasets or checkpoints.

## Included Models

The final main table contains four models:

| Person | Backbone | Fusion | Selected weight |
|---|---|---|---|
| person1 | FPN | BiLocal | `.pkl` |
| person1 | FPN | AsymBi | `.pkl` |
| person3 | UNet | AsymBi | `asymbi.pt` |
| person4 | UNet | BiGlobal | `.pkl` |

## Main Table

Main quantitative table:

* `tables/final_metrics_all_available.csv`

person1-only table:

* `tables/final_metrics_person1.csv`

Unavailable planned experiments:

* `tables/unavailable_experiments_note.csv`

## Figures

Metric figures are stored in `figures/`:

* `all_available_iou_comparison.png`
* `all_available_niou_comparison.png`
* `all_available_pd_comparison.png`
* `all_available_fa_comparison.png`
* `person1_iou_comparison.png`
* `person1_niou_comparison.png`

## Visualizations

Full generated qualitative results are stored in `visualizations/`:

* `visualizations/FPN_BiLocal/`
* `visualizations/FPN_AsymBi/`
* `visualizations/UNet_AsymBi/`
* `visualizations/UNet_BiGlobal/`

These folders may contain many generated images and are normally excluded from GitHub by `.gitignore`.

## Bad Cases

Bad-case candidates are stored in `bad_cases/`.

The metadata table is:

* `bad_cases/bad_case_table.csv`

## Gray Surface

The 3D gray response examples are stored in `gray_surface/`. These images describe the SIRST infrared small-target data itself and are not model-dependent.

## Selected Examples

Small representative images for GitHub or PPT preview are stored in:

* `selected_examples/`

The selected set is intentionally small to keep the repository lightweight.

## Report Snippets

Copy-ready report fragments are stored in `report_snippets/`:

* `experiment_setup.md`
* `results_analysis_person1.md`
* `results_analysis_all_available.md`
* `limitations.md`
* `metric_definitions.md`

## Notes

The final SIRST test split is `data/sirst/idx_427/test.txt` with 640 images and 0 train/test overlap. In the current ACM code, validation mode reads this same split, so this should be described clearly in reports.

The current experiment pack does not include datasets or checkpoints. GitHub should contain only code, documentation, tables, figures, and a small number of selected example images.

YOLO mAP50, PConv ablations, SD Loss ablations, and NUAA cross-dataset testing are marked as N/A for the current ACM segmentation main table.
