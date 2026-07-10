# External Results Sources

This document records the local source folders used to integrate person3 and person4 results into the final experiment package.

## person3

Source folder:

`C:\Users\bbqsa\Desktop\红外检测\infrared_detection_inference_v2-xcy34`

Selected weight:

`inference_package\checkpoint\asymbi.pt`

Final metrics on `data/sirst/idx_427/test.txt`:

| Backbone | Fusion | IoU | nIoU |
|---|---|---:|---:|
| UNet | AsymBi | 0.3262435852 | 0.3772964333 |

Note: the alternative `Epoch- 30_IoU-0.2551_nIoU-0.3876.pkl` was not selected for the main table because `asymbi.pt` has higher final-test IoU. Do not mix `.pt` and `.pkl` metrics.

## person4

Source folder:

`C:\Users\bbqsa\Desktop\红外检测\2026-07-09-23-56-57_UNet_BiGlobal-hk`

Selected weight:

`checkpoint\Epoch- 29_IoU-0.3884_nIoU-0.4837.pkl`

Final metrics on `data/sirst/idx_427/test.txt`:

| Backbone | Fusion | IoU | nIoU |
|---|---|---:|---:|
| UNet | BiGlobal | 0.3883507577 | 0.4837368390 |

## Repository Policy

These external folders are local result sources only. They are not uploaded to GitHub.

The GitHub repository stores only the integrated code, documents, tables, figures, report snippets, and a small number of selected example images.

Do not upload external result folders, datasets, checkpoints, full logs, compressed packages, or full visualization directories.
