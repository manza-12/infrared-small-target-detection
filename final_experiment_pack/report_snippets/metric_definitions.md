# Metric Definitions

IoU: intersection-over-union between the thresholded prediction mask and the ground-truth mask, computed by the ACM segmentation evaluator.

nIoU: sample-wise normalized IoU used by the ACM codebase, computed over thresholded binary masks and aggregated over the test split.

Pd: image-level probability of detection. A target image is counted as detected if the prediction mask and the GT mask have at least one overlapping foreground pixel.

Fa: false alarm value. In this package, `Fa = false_alarm_pixels / total_images`, where false alarm pixels are predicted foreground pixels outside the GT foreground region.

mAP50: a detection-box metric at IoU threshold 0.50. It is not used in the current ACM segmentation main table.
