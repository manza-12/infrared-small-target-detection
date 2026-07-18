# Metric Definitions

- IoU: global segmentation intersection over union between predicted target pixels and ground-truth target pixels.
- nIoU: sample-wise normalized IoU used by the ACM evaluation code.
- Pd: image-level detection probability, computed as `detected target images / target images`.
- Fa: false alarm pixels per image, computed as `false_alarm_pixels / total_images`.

mAP50 is not used in the current ACM segmentation main table. It belongs to the YOLO detection-box evaluation line and should not be mixed directly with ACM segmentation-mask metrics.
