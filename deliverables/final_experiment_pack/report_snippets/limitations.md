# Limitations

The final test and the validation stage in the current ACM code both use `data/sirst/idx_427/test.txt`. Therefore, these results should be described as a comparison on a fixed SIRST split, not as an external independent generalization experiment.

The current completed study is a backbone and fusion-strategy ablation for ACM segmentation. PConv, SD Loss, NUAA cross-dataset generalization, YOLO detection mAP50, and YOLO box-level detection results are not included in the current six-model ACM main table.

The repository does not include datasets, checkpoints, full prediction outputs, or full training logs, so reproducing the numerical results requires placing the corresponding files back into the ignored local directories.
