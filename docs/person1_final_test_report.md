# Person1 Final Test Report

## Purpose

Evaluate the two trained ACM/person1 checkpoints on the final test split without further training.

## Test Set Source

* Test split file: `data/sirst/idx_427/test.txt`
* Test images directory: `data/sirst/images`
* Test masks directory: `data/sirst/masks`
* Number of test images: 640
* Number of test masks: 640
* Train/test overlap: 0 names overlapped between `trainval.txt` and `test.txt`

Note: in the current ACM dataset class, `mode='val'` reads `idx_427/test.txt`. The final test metrics below therefore use the explicit `test.txt` split through the same validation transform and metric logic as training.

## Evaluation Commands

### Smoke Test

```powershell
& "E:\tools\miniconda3\Library\bin\conda.bat" run --no-capture-output -n geoai-yolo python tools\evaluate_acm_checkpoint.py --backbone-mode FPN --fuse-mode BiLocal --checkpoint "experiments\\training_runs\\2026-07-10-01-38-03_FPN_BiLocal\checkpoint\Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl" --data-root data\sirst --split test --batch-size 8 --output-dir experiments\\final_test_outputs\\person1\FPN_BiLocal --max-batches 1
```

### FPN + BiLocal

```powershell
& "E:\tools\miniconda3\Library\bin\conda.bat" run --no-capture-output -n geoai-yolo python tools\evaluate_acm_checkpoint.py --backbone-mode FPN --fuse-mode BiLocal --checkpoint "experiments\\training_runs\\2026-07-10-01-38-03_FPN_BiLocal\checkpoint\Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl" --data-root data\sirst --split test --batch-size 8 --output-dir experiments\\final_test_outputs\\person1\FPN_BiLocal
```

### FPN + AsymBi

```powershell
& "E:\tools\miniconda3\Library\bin\conda.bat" run --no-capture-output -n geoai-yolo python tools\evaluate_acm_checkpoint.py --backbone-mode FPN --fuse-mode AsymBi --checkpoint "experiments\\training_runs\\2026-07-10-02-30-15_FPN_AsymBi\checkpoint\Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl" --data-root data\sirst --split test --batch-size 8 --output-dir experiments\\final_test_outputs\\person1\FPN_AsymBi
```

## Checkpoints

* FPN + BiLocal: `experiments/training_runs/2026-07-10-01-38-03_FPN_BiLocal/checkpoint/Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl`
* FPN + AsymBi: `experiments/training_runs/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl`
* FPN + AsymBi best-nIoU checkpoint also exists but was not evaluated: `experiments/training_runs/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 22_IoU-0.3752_nIoU-0.4941.pkl`

## Metrics

| Backbone | Fusion | Train/Val Best IoU | Train/Val Best nIoU | Final Test IoU | Final Test nIoU | Notes |
|---|---|---:|---:|---:|---:|---|
| FPN | BiLocal | 0.26645 | 0.30931 | 0.26644892435619305 | 0.3093132385053442 | Best IoU and best nIoU came from the same epoch/checkpoint. |
| FPN | AsymBi | 0.40594 | 0.49409 | 0.40594209225945366 | 0.48558731667721544 | Final test used the epoch 26 best-IoU checkpoint. Best nIoU was at epoch 22 and was not separately tested. |

## Outputs

* FPN + BiLocal output directory: `experiments/final_test_outputs/person1/FPN_BiLocal`
* FPN + AsymBi output directory: `experiments/final_test_outputs/person1/FPN_AsymBi`
* Each output directory contains `test_log.txt`, `metrics.json`, `metrics.csv`, and `config_used.txt`.
* Prediction masks/visualizations were not saved in this first-stage evaluator; only metrics were produced.

## Observations

* No evaluation failure occurred.
* The final test metrics match the training validation metrics because the ACM code uses `idx_427/test.txt` as its validation split.
* The PowerShell output includes a `NativeCommandError` wrapper around `conda.bat` stderr/progress text, but each evaluation command returned `EXIT_CODE=0` and produced metrics files.

## Recommendation

If the group wants to report the highest nIoU model for FPN + AsymBi, evaluate the epoch 22 checkpoint separately after confirmation. For the current requested run, the epoch 26 best-IoU checkpoint is the selected final-test checkpoint.
