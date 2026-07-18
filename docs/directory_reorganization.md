# Directory Reorganization

## Purpose

This change reorganizes repository hierarchy so code, experiment outputs, and report deliverables have clearer containment. It does not reduce repository contents and does not move files to GitHub Releases.

## Before

```text
.
|-- result/
|-- final_test_outputs/
|-- background_runs/
`-- final_experiment_pack/
```

## After

```text
.
|-- experiments/
|   |-- training_runs/
|   |-- final_test_outputs/
|   `-- background_runs/
`-- deliverables/
    `-- final_experiment_pack/
```

## Path Mapping

| Old path | New path | Content changed |
|---|---|---|
| `result/` | `experiments/training_runs/` | false |
| `final_test_outputs/` | `experiments/final_test_outputs/` | false |
| `background_runs/` | `experiments/background_runs/` | false |
| `final_experiment_pack/` | `deliverables/final_experiment_pack/` | false |

## Integrity Notes

- No files were intentionally deleted.
- Checkpoint contents were not intentionally modified.
- Training log contents were not intentionally modified.
- Prediction images and gray-surface images were not intentionally modified.
- Historical logs may still contain old root-level paths. Those paths describe the state before this reorganization and are intentionally left unchanged.
- Numeric experiment metrics in CSV files were not intentionally changed. Only path strings were updated where they are used for navigation or execution.

## Running With New Paths

Training entry point remains:

```powershell
python run.py --person 2 --epochs 30 --batch-size 8
```

Evaluation checkpoint paths now use `experiments/training_runs/`:

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

Visualization outputs now use `deliverables/final_experiment_pack/`:

```powershell
python tools/visualize_acm_predictions.py `
  --backbone-mode FPN `
  --fuse-mode BiGlobal `
  --checkpoint "experiments/training_runs/2026-07-10-11-40-36_FPN_BiGlobal/checkpoint/Epoch- 29_IoU-0.3661_nIoU-0.4457.pkl" `
  --data-root data/sirst `
  --split-file idx_427/test.txt `
  --output-dir deliverables/final_experiment_pack/visualizations/FPN_BiGlobal
```