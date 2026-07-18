# Person1 Result Analysis

| Backbone | Fusion | Final Test IoU | Final Test nIoU | Notes |
|---|---|---:|---:|---|
| FPN | BiLocal | 0.2664489244 | 0.3093132385 | Selected checkpoint is epoch 29. |
| FPN | AsymBi | 0.4059420923 | 0.4855873167 | Selected checkpoint is epoch 26 by best IoU. |

Within the FPN backbone setting, FPN+AsymBi is clearly better than FPN+BiLocal on both IoU and nIoU in the current person1 result set.

This conclusion is limited to the current ACM/FPN segmentation comparison and should not be generalized to all model families. For FPN+AsymBi, the best-IoU checkpoint and best-nIoU checkpoint are different: epoch 26 is selected by best IoU, while epoch 22 has the highest nIoU reported during training.
