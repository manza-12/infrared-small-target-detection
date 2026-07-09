# Limitations

The current deliverable is primarily based on person1, because no completed person2/person3/person4 final-test outputs were found in the scanned workspace.

Planned YOLO/PConv/SD Loss/mAP50 and cross-dataset generalization experiments are marked as N/A because the current workspace does not contain the required checkpoints, data, or final evaluation outputs.

The ACM segmentation line and YOLO detection-box line should not be merged into one metric table as if their metrics were directly equivalent. ACM uses IoU/nIoU over segmentation masks, while YOLO mAP50 is a detection-box metric.
