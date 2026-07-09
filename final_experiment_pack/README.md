# Final Experiment Pack

This folder collects the currently deliverable materials for Experiment A and Experiment B based on available ACM/person1 results. No training was performed while creating this package.

## Tables

* `tables/final_metrics_person1.csv`: final person1 IoU/nIoU table, with Pd/Fa if available.
* `tables/final_metrics_all_available.csv`: all scanned configurations with usable final-test results. At present this is person1 only.
* `tables/unavailable_experiments_note.csv`: planned but unavailable experiments marked as N/A.

## Figures

* `figures/person1_iou_comparison.png`
* `figures/person1_niou_comparison.png`
* `figures/all_available_iou_comparison.png`
* `figures/all_available_niou_comparison.png`

## Visualizations

* `visualizations/FPN_BiLocal/`: original / GT / prediction triplets for FPN+BiLocal.
* `visualizations/FPN_AsymBi/`: original / GT / prediction triplets for FPN+AsymBi.

## Bad Cases

* `bad_cases/`: low-IoU candidate triplets.
* `bad_cases/bad_case_table.csv`: candidate metadata.

## Gray Surface

* `gray_surface/`: 3D gray response surfaces and matching 2D patches. Generated cases: 10.

## Report Snippets

* `report_snippets/experiment_setup.md`
* `report_snippets/results_analysis_person1.md`
* `report_snippets/limitations.md`
* `report_snippets/metric_definitions.md`

## Completed and N/A Items

Completed: person1 FPN+BiLocal and FPN+AsymBi on SIRST `test.txt`.

N/A: PConv k=3/4 ablation, SD Loss alpha ablation, SIRST to NUAA generalization, and YOLO mAP50 comparison, because the current ACM segmentation workspace does not contain the required checkpoints or data.

## How to Use in the Paper

Use `tables/final_metrics_person1.csv` for the quantitative table, `figures/` for metric bar charts, `visualizations/` for qualitative segmentation examples, `bad_cases/` for failure analysis, and `gray_surface/` for infrared small-target gray response illustration.
