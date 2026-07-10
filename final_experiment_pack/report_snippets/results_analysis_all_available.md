# All Available Result Analysis

The final available ACM segmentation comparison includes four models on SIRST `data/sirst/idx_427/test.txt`:

| Person | Backbone | Fusion | IoU | nIoU | Pd | Fa |
|---|---|---|---:|---:|---:|---:|
| person1 | FPN | BiLocal | 0.2664489244 | 0.3093132385 | 0.987500 | 704.678125 |
| person1 | FPN | AsymBi | 0.4059420923 | 0.4855873167 | 0.990625 | 360.070313 |
| person3 | UNet | AsymBi | 0.3262435852 | 0.3772964333 | 0.995313 | 530.640625 |
| person4 | UNet | BiGlobal | 0.3883507577 | 0.4837368390 | 0.993750 | 389.137500 |

Within the FPN branch, FPN+AsymBi is clearly better than FPN+BiLocal on IoU and nIoU.

Within the UNet branch, UNet+BiGlobal is better than UNet+AsymBi, especially on nIoU.

Across the available models, FPN+AsymBi achieves the highest IoU. FPN+AsymBi and UNet+BiGlobal have very close nIoU values, which suggests that UNet+BiGlobal is a strong alternative under the current ACM segmentation setting.

Weight selection notes:

* person3 uses `asymbi.pt` because it has higher final-test IoU than the `.pkl` candidate. The `.pkl` candidate has slightly higher nIoU but much lower IoU.
* person4 uses the `.pkl` checkpoint from the complete result directory because it was verified by the same evaluator and matches the filename metrics.
* `.pt` and `.pkl` metrics are not mixed.

Limitations: person2 is not included. PConv, SD Loss, NUAA cross-dataset testing, and YOLO mAP50 are not included in the current final main table.
