# Metric Definitions

IoU: intersection-over-union between the thresholded prediction mask and the ground-truth mask, accumulated by the ACM segmentation evaluator.

nIoU: sample-wise normalized IoU used by the ACM codebase, computed over thresholded binary masks for each image and then aggregated.

Pd: image-level probability of detection. In this package, a target image is counted as detected if the prediction mask and GT mask have at least one overlapping foreground pixel.

Fa: false alarm value. In this package, `Fa = false_alarm_pixels / total_images`, where false alarm pixels are predicted foreground pixels outside the GT foreground region.

mAP50: a YOLO detection-box metric at IoU threshold 0.50. It is not applicable to the current ACM segmentation line and is therefore marked as N/A.
