# Limitations

The current final comparison contains four available ACM segmentation models: FPN+BiLocal, FPN+AsymBi, UNet+AsymBi, and UNet+BiGlobal.

person2 is not included because no confirmed final-test result was available at the time of packaging.

The planned PConv ablation, SD Loss ablation, cross-dataset generalization, and YOLO mAP50 comparison are marked as N/A because the current workspace does not contain the required final checkpoints, prepared data, or compatible metric pipeline.

The final test uses the same `data/sirst/idx_427/test.txt` split that the ACM validation mode reads. Therefore, these numbers should be reported as final SIRST test-split results, not as external independent generalization results.

The ACM segmentation line and YOLO detection-box line should not be merged into one metric table. ACM reports mask metrics such as IoU/nIoU/Pd/Fa, while YOLO mAP50 is a detection-box metric.
