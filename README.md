# Infrared Small Target Detection / Segmentation Course Project

## Overview

This repository contains the course project code and experiment materials for infrared small target detection and segmentation. The completed experiment line is the ACM segmentation line on the fixed SIRST `idx_427` split.

The repository includes code, documentation, tables, figures, selected examples, checkpoints, training logs, full prediction visualizations, and final test outputs so that teammates can review the results directly on GitHub.

The dataset itself is not included. To reproduce evaluation or training, place the SIRST data under `data/sirst/` following the expected structure.

## Completed Experiments

The ACM main table covers a complete 2 backbone x 3 fusion strategy matrix:

- FPN + BiLocal
- FPN + AsymBi
- FPN + BiGlobal
- UNet + BiLocal
- UNet + AsymBi
- UNet + BiGlobal

The unified test split is `data/sirst/idx_427/test.txt`, containing 640 images. In the current ACM code, validation mode also reads this test split, so final test metrics are expected to match or closely match validation metrics. This is a fixed-split comparison, not an external generalization test.

## Main Results

Fa is defined as `false_alarm_pixels / total_images`.

| Person | Backbone | Fusion | IoU | nIoU | Pd | Fa |
|---|---|---|---:|---:|---:|---:|
| person1 | FPN | BiLocal | 0.2664489244 | 0.3093132385 | 0.9875000 | 704.678125 |
| person1 | FPN | AsymBi | 0.4059420923 | 0.4855873167 | 0.9906250 | 360.070312 |
| person2 | FPN | BiGlobal | 0.3660601796 | 0.4457140235 | 0.9890625 | 440.821875 |
| person2 | UNet | BiLocal | 0.3772599337 | 0.4348144315 | 0.9859375 | 415.790625 |
| person3 | UNet | AsymBi | 0.3262435852 | 0.3772964333 | 0.9953125 | 530.640625 |
| person4 | UNet | BiGlobal | 0.3883507577 | 0.4837368390 | 0.9937500 | 389.137500 |

## Repository Structure

```text
.
|-- acm/                                  # ACM segmentation source code
|-- models/                               # Model-related code
|-- tools/                                # Evaluation, Pd/Fa, visualization, and packaging scripts
|-- scripts/                              # Helper scripts
|-- tests/                                # Unit tests
|-- experiments/
|   |-- training_runs/                    # Full training results and checkpoints
|   |-- final_test_outputs/               # Six-model final test logs and metrics
|   `-- background_runs/                  # Background training commands and logs
|-- deliverables/
|   `-- final_experiment_pack/
|       |-- tables/                       # Main table and ablation tables
|       |-- figures/                      # Metric comparison figures
|       |-- report_snippets/              # Report-ready text snippets
|       |-- selected_examples/            # Representative preview images
|       |-- visualizations/               # Full prediction triplets
|       |-- bad_cases/                    # Failure cases
|       |-- gray_surface/                 # 3D gray response plots
|       `-- logs/                         # Pack generation logs
|-- docs/                                 # Notes, records, and migration documentation
|-- run.py                                # Person-based training entry point
|-- infer.py
|-- setup_data.py
|-- requirements.txt
`-- .gitignore
```

This reorganization changes only the directory hierarchy. It does not reduce repository contents. The repository still includes full checkpoints, full training logs, final test outputs, and full prediction images.

## Included Artifacts

- ACM pretrained/reference checkpoints under `acm/params/`
- Experiment checkpoints under `experiments/training_runs/`
- Final test outputs under `experiments/final_test_outputs/`
- Background training logs under `experiments/background_runs/`
- Full prediction visualizations under `deliverables/final_experiment_pack/visualizations/`
- Bad-case images and metadata under `deliverables/final_experiment_pack/bad_cases/`
- Gray-surface examples under `deliverables/final_experiment_pack/gray_surface/`
- Packaging logs under `deliverables/final_experiment_pack/logs/`
- Summary tables under `deliverables/final_experiment_pack/tables/`
- Report snippets under `deliverables/final_experiment_pack/report_snippets/`

The repository still excludes:

- `data/`
- `datasets/`
- temporary archives such as `.zip`, `.tar.gz`, `.7z`
- Python caches and editor files

## Environment

Local experiment environment:

- Python 3.11
- torch 2.6.0 + cu124
- CUDA available
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU 8GB

The minimal Python dependency list is in `requirements.txt`. For reproduction, install `torch` and `torchvision` versions compatible with the local CUDA runtime.

## Evaluation Examples

FPN + BiGlobal final test:

```powershell
python tools/evaluate_acm_checkpoint.py `
  --backbone-mode FPN `
  --fuse-mode BiGlobal `
  --checkpoint "experiments/training_runs/2026-07-10-11-40-36_FPN_BiGlobal/checkpoint/Epoch- 29_IoU-0.3661_nIoU-0.4457.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir experiments/final_test_outputs/person2/FPN_BiGlobal_selected
```

UNet + BiLocal Pd/Fa evaluation:

```powershell
python tools/evaluate_acm_pd_fa.py `
  --backbone-mode UNet `
  --fuse-mode BiLocal `
  --checkpoint "experiments/training_runs/2026-07-10-12-58-28_UNet_BiLocal/checkpoint/Epoch- 30_IoU-0.3773_nIoU-0.4348.pkl" `
  --data-root data/sirst `
  --split test `
  --batch-size 8 `
  --output-dir experiments/final_test_outputs/person2/UNet_BiLocal_selected
```

Generate prediction triplets:

```powershell
python tools/visualize_acm_predictions.py `
  --backbone-mode FPN `
  --fuse-mode BiGlobal `
  --checkpoint "experiments/training_runs/2026-07-10-11-40-36_FPN_BiGlobal/checkpoint/Epoch- 29_IoU-0.3661_nIoU-0.4457.pkl" `
  --data-root data/sirst `
  --split-file idx_427/test.txt `
  --output-dir deliverables/final_experiment_pack/visualizations/FPN_BiGlobal `
  --max-images 50
```

## Deliverables

The main deliverable package is `deliverables/final_experiment_pack/`. Start with `deliverables/final_experiment_pack/INDEX.md` for a compact navigation page.

## Limitations

- The dataset is not included in this repository.
- The current completed study is an ACM segmentation backbone/fusion ablation.
- PConv k=3/4, SD Loss, SIRST-to-NUAA generalization, and YOLO mAP50 are not part of the current ACM six-model main table.
- ACM segmentation metrics IoU/nIoU/Pd/Fa should not be mixed directly with YOLO detection mAP50.