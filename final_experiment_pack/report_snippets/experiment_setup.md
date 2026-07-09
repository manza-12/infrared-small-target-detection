# Experiment Setup

Dataset: SIRST.

Test split: `data/sirst/idx_427/test.txt`.

Number of test images: 640.

Train/test overlap: 0 names overlapped between `trainval.txt` and `test.txt`.

Model line: ACM segmentation line.

Person1 configurations: FPN+BiLocal and FPN+AsymBi.

Metrics: IoU and nIoU are the primary segmentation metrics. Pd/Fa are reported when computed from thresholded prediction masks; mAP50 is not used for the ACM segmentation line.
