# Experiment Setup

The experiments use the SIRST `idx_427` split. The final test split is `data/sirst/idx_427/test.txt` and contains 640 test images. The train/test overlap is 0 according to the prepared split files.

The completed ACM segmentation comparison covers a full 2 backbones x 3 fusion strategies matrix:

- Backbones: FPN, UNet
- Fusion strategies: BiLocal, AsymBi, BiGlobal
- Total models: six ACM segmentation models

All models are evaluated on the same SIRST test split with IoU, nIoU, Pd, and Fa. Pd is image-level target detection probability. Fa is defined as `false_alarm_pixels / total_images`.

mAP50 is not used in the current ACM segmentation main table because it is a detection-box metric for the YOLO line, not a segmentation-mask metric for these ACM experiments.
