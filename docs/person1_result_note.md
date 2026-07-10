# Person1 Formal Training Record

## Environment

* Conda environment: `geoai-yolo`
* Python: 3.11.15
* Torch: 2.6.0+cu124
* CUDA: True
* GPU: NVIDIA GeForce RTX 4060 Laptop GPU

## Data

* Dataset: SIRST
* Train: 2114
* Val: 640

## Task Configurations

* Config 1: FPN + BiLocal
* Config 2: FPN + AsymBi

## Training Settings

* Epochs: 30
* Batch size: 8
* Launch command:

```powershell
& "E:\tools\miniconda3\Library\bin\conda.bat" run --no-capture-output -n geoai-yolo python run.py --person 1 --epochs 30 --batch-size 8 2>&1 | Tee-Object -FilePath person1_full_train_log.txt
```

## Output Locations

* Result root: `result`
* FPN + BiLocal result directory: `result/2026-07-10-01-38-03_FPN_BiLocal`
* FPN + BiLocal selected checkpoint: `result/2026-07-10-01-38-03_FPN_BiLocal/checkpoint/Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl`
* FPN + AsymBi result directory: `result/2026-07-10-02-30-15_FPN_AsymBi`
* FPN + AsymBi selected checkpoint by best IoU: `result/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl`
* FPN + AsymBi best nIoU checkpoint: `result/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 22_IoU-0.3752_nIoU-0.4941.pkl`
* Training log: `person1_full_train_log.txt`

## Metrics

| Backbone | Fusion | Best IoU | Best IoU Epoch | Best nIoU | Best nIoU Epoch | Final Epoch IoU | Final Epoch nIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| FPN | BiLocal | 0.26645 | 29 | 0.30931 | 29 | 0.22930 | 0.28537 |
| FPN | AsymBi | 0.40594 | 26 | 0.49409 | 22 | 0.37091 | 0.45665 |

## Notes

* Success: yes
* Warning: PowerShell logged a `NativeCommandError` wrapper around `conda.bat` stderr output, but the command completed with `EXIT_CODE=0`, and both configurations printed `DONE`.
* Checkpoint selection: BiLocal has best IoU and best nIoU in the same checkpoint. AsymBi has best IoU at epoch 26 and best nIoU at epoch 22, so report both values and specify which checkpoint is selected.
* Need rerun: no immediate rerun needed for person1 result reporting.
