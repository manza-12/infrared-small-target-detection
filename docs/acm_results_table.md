# ACM Person1 Results

## Summary Table

| Backbone | Fusion | Epochs | Batch Size | Best IoU | Best IoU Epoch | Best nIoU | Best nIoU Epoch | Selected Checkpoint | Status | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| FPN | BiLocal | 30 | 8 | 0.26645 | 29 | 0.30931 | 29 | `experiments/training_runs/2026-07-10-01-38-03_FPN_BiLocal/checkpoint/Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl` | success | Best IoU and best nIoU are from the same epoch/checkpoint. |
| FPN | AsymBi | 30 | 8 | 0.40594 | 26 | 0.49409 | 22 | `experiments/training_runs/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl` | success | Selected checkpoint is by best IoU. Best nIoU is higher at epoch 22: `Epoch- 22_IoU-0.3752_nIoU-0.4941.pkl`. |

## Per-Epoch Validation Metrics

### FPN + BiLocal

| Epoch | IoU | nIoU |
|---:|---:|---:|
| 1 | 0.002334 | 0.025779 |
| 2 | 0.003059 | 0.015953 |
| 3 | 0.065374 | 0.092155 |
| 4 | 0.038273 | 0.070100 |
| 5 | 0.059955 | 0.084320 |
| 6 | 0.062333 | 0.094397 |
| 7 | 0.090063 | 0.120439 |
| 8 | 0.037952 | 0.082830 |
| 9 | 0.108481 | 0.152654 |
| 10 | 0.073658 | 0.111580 |
| 11 | 0.157820 | 0.174766 |
| 12 | 0.090589 | 0.131666 |
| 13 | 0.080854 | 0.119268 |
| 14 | 0.104951 | 0.162460 |
| 15 | 0.114926 | 0.146809 |
| 16 | 0.131764 | 0.187252 |
| 17 | 0.180484 | 0.186248 |
| 18 | 0.115993 | 0.194798 |
| 19 | 0.125071 | 0.199978 |
| 20 | 0.162667 | 0.217603 |
| 21 | 0.140380 | 0.201829 |
| 22 | 0.152747 | 0.238429 |
| 23 | 0.175133 | 0.222096 |
| 24 | 0.183256 | 0.239739 |
| 25 | 0.165846 | 0.229504 |
| 26 | 0.213791 | 0.244742 |
| 27 | 0.186947 | 0.255736 |
| 28 | 0.171832 | 0.258119 |
| 29 | 0.266449 | 0.309313 |
| 30 | 0.229295 | 0.285374 |

### FPN + AsymBi

| Epoch | IoU | nIoU |
|---:|---:|---:|
| 1 | 0.064472 | 0.101474 |
| 2 | 0.086137 | 0.102187 |
| 3 | 0.079948 | 0.104136 |
| 4 | 0.043010 | 0.086897 |
| 5 | 0.061250 | 0.128670 |
| 6 | 0.090151 | 0.141366 |
| 7 | 0.086362 | 0.185636 |
| 8 | 0.213119 | 0.282938 |
| 9 | 0.179493 | 0.234115 |
| 10 | 0.167754 | 0.273226 |
| 11 | 0.162442 | 0.265895 |
| 12 | 0.241281 | 0.342088 |
| 13 | 0.219607 | 0.301853 |
| 14 | 0.190289 | 0.368563 |
| 15 | 0.293188 | 0.420813 |
| 16 | 0.336047 | 0.436324 |
| 17 | 0.102861 | 0.280420 |
| 18 | 0.259440 | 0.353843 |
| 19 | 0.153164 | 0.343525 |
| 20 | 0.229088 | 0.418848 |
| 21 | 0.175584 | 0.397103 |
| 22 | 0.375157 | 0.494092 |
| 23 | 0.336839 | 0.439202 |
| 24 | 0.371781 | 0.421614 |
| 25 | 0.306037 | 0.414030 |
| 26 | 0.405942 | 0.485587 |
| 27 | 0.335376 | 0.426721 |
| 28 | 0.389320 | 0.465622 |
| 29 | 0.383857 | 0.441830 |
| 30 | 0.370908 | 0.456654 |

## Checkpoint Interpretation

- `FPN + BiLocal`: best IoU and best nIoU both occur at epoch 29, so the selected checkpoint records both best values.
- `FPN + AsymBi`: best IoU occurs at epoch 26 (`IoU=0.405942`, `nIoU=0.485587`), while best nIoU occurs at epoch 22 (`IoU=0.375157`, `nIoU=0.494092`). The `0.49409` value comes from the training log summary and the epoch 22 checkpoint filename, while `0.4856` is the nIoU stored in the epoch 26 best-IoU checkpoint filename.
