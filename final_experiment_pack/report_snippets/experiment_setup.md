# Experiment Setup

Dataset: SIRST.

Test split: `data/sirst/idx_427/test.txt`.

Number of test images: 640.

Train/test overlap: 0 names overlapped between `trainval.txt` and `test.txt`.

Task line: ACM segmentation line. In the current ACM codebase, `mode='val'` reads `idx_427/test.txt`, so final test and training validation use the same split. This should be stated explicitly and should not be interpreted as an external cross-dataset generalization test.

Models included in the final main table:

| Person | Backbone | Fusion | Selected weight type |
|---|---|---|---|
| person1 | FPN | BiLocal | `.pkl` |
| person1 | FPN | AsymBi | `.pkl` |
| person3 | UNet | AsymBi | `.pt` |
| person4 | UNet | BiGlobal | `.pkl` |

Metrics: IoU, nIoU, Pd, and Fa. Fa is defined as `false_alarm_pixels / total_images`.

mAP50 is a detection-box metric and is not applicable to the current ACM segmentation main table.
