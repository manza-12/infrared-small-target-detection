#!/usr/bin/env python3
"""Build final experiment tables, figures, gray surfaces, and report snippets."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = PROJECT_ROOT / "final_experiment_pack"

PERSON1 = [
    {
        "experiment_line": "person1",
        "backbone": "FPN",
        "fusion": "BiLocal",
        "checkpoint": "result/2026-07-10-01-38-03_FPN_BiLocal/checkpoint/Epoch- 29_IoU-0.2664_nIoU-0.3093.pkl",
        "test_set": "data/sirst/idx_427/test.txt",
        "num_test_images": 640,
        "IoU": 0.26644892435619305,
        "nIoU": 0.3093132385053442,
        "mAP50": "N/A",
        "notes": "ACM segmentation line; final test uses data/sirst/idx_427/test.txt.",
    },
    {
        "experiment_line": "person1",
        "backbone": "FPN",
        "fusion": "AsymBi",
        "checkpoint": "result/2026-07-10-02-30-15_FPN_AsymBi/checkpoint/Epoch- 26_IoU-0.4059_nIoU-0.4856.pkl",
        "test_set": "data/sirst/idx_427/test.txt",
        "num_test_images": 640,
        "IoU": 0.40594209225945366,
        "nIoU": 0.48558731667721544,
        "mAP50": "N/A",
        "notes": "ACM segmentation line; selected checkpoint is best IoU. Epoch 22 has best nIoU 0.49409 but is not selected here.",
    },
]


def ensure_dirs():
    for sub in ["tables", "figures", "visualizations", "bad_cases", "gray_surface", "logs", "report_snippets"]:
        (PACK_DIR / sub).mkdir(parents=True, exist_ok=True)


def read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables():
    pd_fa_lookup = {
        "BiLocal": read_json(PROJECT_ROOT / "final_test_outputs/person1/FPN_BiLocal/metrics_pd_fa.json"),
        "AsymBi": read_json(PROJECT_ROOT / "final_test_outputs/person1/FPN_AsymBi/metrics_pd_fa.json"),
    }
    person1_rows = []
    for row in PERSON1:
        pd_fa = pd_fa_lookup.get(row["fusion"], {})
        out = dict(row)
        out["Pd"] = pd_fa.get("Pd", "N/A")
        out["Fa"] = pd_fa.get("Fa", "N/A")
        if out["Pd"] == "N/A" or out["Fa"] == "N/A":
            out["notes"] += " current ACM evaluator reports IoU/nIoU only; Pd/Fa not computed in this stage."
        else:
            out["notes"] += " Pd/Fa computed from thresholded prediction masks; Fa=false_alarm_pixels/total_images."
        person1_rows.append(out)

    metric_fields = [
        "experiment_line", "backbone", "fusion", "checkpoint", "test_set", "num_test_images",
        "IoU", "nIoU", "mAP50", "Pd", "Fa", "notes",
    ]
    write_csv(PACK_DIR / "tables/final_metrics_person1.csv", person1_rows, metric_fields)

    all_rows = []
    for row in person1_rows:
        all_row = dict(row)
        all_row["status"] = "final_test_done"
        all_rows.append(all_row)
    write_csv(PACK_DIR / "tables/final_metrics_all_available.csv", all_rows, metric_fields + ["status"])

    unavailable = [
        {
            "experiment": "PConv k=3 vs k=4 ablation",
            "status": "N/A",
            "reason": "No available ACM/Yolo checkpoint for this item in current workspace.",
            "what_is_needed_to_run": "Model code, trained checkpoints, and a confirmed evaluation protocol for PConv variants.",
        },
        {
            "experiment": "SD Loss alpha=0.3/0.5/0.7 ablation",
            "status": "N/A",
            "reason": "Current ACM segmentation line does not include this loss ablation result.",
            "what_is_needed_to_run": "Training outputs or checkpoints for each SD Loss alpha setting.",
        },
        {
            "experiment": "SIRST to NUAA cross-dataset generalization",
            "status": "N/A",
            "reason": "No NUAA or NUAA-SIRST test data directory was found in the current workspace scan.",
            "what_is_needed_to_run": "NUAA dataset prepared with a documented split and compatible masks or labels.",
        },
        {
            "experiment": "YOLO mAP50 comparison",
            "status": "N/A",
            "reason": "Current completed results are from ACM segmentation checkpoints; YOLO detection checkpoints/results are not available for this comparison.",
            "what_is_needed_to_run": "YOLO prediction/results.csv files or trained YOLO checkpoints evaluated on the same data split.",
        },
    ]
    write_csv(
        PACK_DIR / "tables/unavailable_experiments_note.csv",
        unavailable,
        ["experiment", "status", "reason", "what_is_needed_to_run"],
    )


def plot_metric(metric, filename, title):
    labels = [f"{row['backbone']}+{row['fusion']}" for row in PERSON1]
    values = [row[metric] for row in PERSON1]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C78A8", "#F58518"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PACK_DIR / "figures" / filename, dpi=180)
    plt.close(fig)


def write_figures():
    plot_metric("IoU", "person1_iou_comparison.png", "ACM segmentation line - Person1 IoU")
    plot_metric("nIoU", "person1_niou_comparison.png", "ACM segmentation line - Person1 nIoU")
    plot_metric("IoU", "all_available_iou_comparison.png", "ACM segmentation line - All Available IoU")
    plot_metric("nIoU", "all_available_niou_comparison.png", "ACM segmentation line - All Available nIoU")


def crop_patch(arr, center_y, center_x, size):
    half = size // 2
    h, w = arr.shape[:2]
    y0 = max(0, center_y - half)
    y1 = min(h, center_y + half)
    x0 = max(0, center_x - half)
    x1 = min(w, center_x + half)
    return arr[y0:y1, x0:x1]


def write_gray_surfaces(max_cases=10, patch_size=40):
    data_root = PROJECT_ROOT / "data/sirst"
    names = [line.strip() for line in (data_root / "idx_427/test.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    saved = 0
    for name in names:
        img_path = data_root / "images" / f"{name}.png"
        mask_path = data_root / "masks" / f"{name}_pixels0.png"
        if not img_path.exists() or not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        gray = np.array(Image.open(img_path).convert("L"))
        patch = crop_patch(gray, cy, cx, patch_size)
        if patch.size == 0:
            continue

        saved += 1
        patch_img = Image.fromarray(patch)
        patch_img.save(PACK_DIR / "gray_surface" / f"case_{saved:02d}_{name}_patch.png")

        yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, patch.astype(float), cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"Infrared Gray Surface - {name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Gray")
        fig.tight_layout()
        fig.savefig(PACK_DIR / "gray_surface" / f"case_{saved:02d}_{name}_surface.png", dpi=180)
        plt.close(fig)
        if saved >= max_cases:
            break
    return saved


def write_snippets(gray_count):
    setup = """# Experiment Setup

Dataset: SIRST.

Test split: `data/sirst/idx_427/test.txt`.

Number of test images: 640.

Train/test overlap: 0 names overlapped between `trainval.txt` and `test.txt`.

Model line: ACM segmentation line.

Person1 configurations: FPN+BiLocal and FPN+AsymBi.

Metrics: IoU and nIoU are the primary segmentation metrics. Pd/Fa are reported when computed from thresholded prediction masks; mAP50 is not used for the ACM segmentation line.
"""
    results = """# Person1 Result Analysis

| Backbone | Fusion | Final Test IoU | Final Test nIoU | Notes |
|---|---|---:|---:|---|
| FPN | BiLocal | 0.2664489244 | 0.3093132385 | Selected checkpoint is epoch 29. |
| FPN | AsymBi | 0.4059420923 | 0.4855873167 | Selected checkpoint is epoch 26 by best IoU. |

Within the FPN backbone setting, FPN+AsymBi is clearly better than FPN+BiLocal on both IoU and nIoU in the current person1 result set.

This conclusion is limited to the current ACM/FPN segmentation comparison and should not be generalized to all model families. For FPN+AsymBi, the best-IoU checkpoint and best-nIoU checkpoint are different: epoch 26 is selected by best IoU, while epoch 22 has the highest nIoU reported during training.
"""
    limitations = """# Limitations

The current deliverable is primarily based on person1, because no completed person2/person3/person4 final-test outputs were found in the scanned workspace.

Planned YOLO/PConv/SD Loss/mAP50 and cross-dataset generalization experiments are marked as N/A because the current workspace does not contain the required checkpoints, data, or final evaluation outputs.

The ACM segmentation line and YOLO detection-box line should not be merged into one metric table as if their metrics were directly equivalent. ACM uses IoU/nIoU over segmentation masks, while YOLO mAP50 is a detection-box metric.
"""
    metric_defs = """# Metric Definitions

IoU: intersection-over-union between the thresholded prediction mask and the ground-truth mask, accumulated by the ACM segmentation evaluator.

nIoU: sample-wise normalized IoU used by the ACM codebase, computed over thresholded binary masks for each image and then aggregated.

Pd: image-level probability of detection. In this package, a target image is counted as detected if the prediction mask and GT mask have at least one overlapping foreground pixel.

Fa: false alarm value. In this package, `Fa = false_alarm_pixels / total_images`, where false alarm pixels are predicted foreground pixels outside the GT foreground region.

mAP50: a YOLO detection-box metric at IoU threshold 0.50. It is not applicable to the current ACM segmentation line and is therefore marked as N/A.
"""
    (PACK_DIR / "report_snippets/experiment_setup.md").write_text(setup, encoding="utf-8")
    (PACK_DIR / "report_snippets/results_analysis_person1.md").write_text(results, encoding="utf-8")
    (PACK_DIR / "report_snippets/limitations.md").write_text(limitations, encoding="utf-8")
    (PACK_DIR / "report_snippets/metric_definitions.md").write_text(metric_defs, encoding="utf-8")

    readme = f"""# Final Experiment Pack

This folder collects the currently deliverable materials for Experiment A and Experiment B based on available ACM/person1 results. No training was performed while creating this package.

## Tables

* `tables/final_metrics_person1.csv`: final person1 IoU/nIoU table, with Pd/Fa if available.
* `tables/final_metrics_all_available.csv`: all scanned configurations with usable final-test results. At present this is person1 only.
* `tables/unavailable_experiments_note.csv`: planned but unavailable experiments marked as N/A.

## Figures

* `figures/person1_iou_comparison.png`
* `figures/person1_niou_comparison.png`
* `figures/all_available_iou_comparison.png`
* `figures/all_available_niou_comparison.png`

## Visualizations

* `visualizations/FPN_BiLocal/`: original / GT / prediction triplets for FPN+BiLocal.
* `visualizations/FPN_AsymBi/`: original / GT / prediction triplets for FPN+AsymBi.

## Bad Cases

* `bad_cases/`: low-IoU candidate triplets.
* `bad_cases/bad_case_table.csv`: candidate metadata.

## Gray Surface

* `gray_surface/`: 3D gray response surfaces and matching 2D patches. Generated cases: {gray_count}.

## Report Snippets

* `report_snippets/experiment_setup.md`
* `report_snippets/results_analysis_person1.md`
* `report_snippets/limitations.md`
* `report_snippets/metric_definitions.md`

## Completed and N/A Items

Completed: person1 FPN+BiLocal and FPN+AsymBi on SIRST `test.txt`.

N/A: PConv k=3/4 ablation, SD Loss alpha ablation, SIRST to NUAA generalization, and YOLO mAP50 comparison, because the current ACM segmentation workspace does not contain the required checkpoints or data.

## How to Use in the Paper

Use `tables/final_metrics_person1.csv` for the quantitative table, `figures/` for metric bar charts, `visualizations/` for qualitative segmentation examples, `bad_cases/` for failure analysis, and `gray_surface/` for infrared small-target gray response illustration.
"""
    (PACK_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_scan_summary():
    lines = [
        "# Scan Summary",
        "",
        "Completed final-test configurations found:",
        "",
        "* person1 FPN+BiLocal: final test metrics available.",
        "* person1 FPN+AsymBi: final test metrics available.",
        "",
        "No completed person2/person3/person4 final-test output directories were found under `final_test_outputs/`.",
        "",
        "No NUAA or NUAA-SIRST dataset directory was found in the current workspace scan.",
        "",
        "No usable YOLO/PConv/SD Loss checkpoint for the requested unavailable experiments was found in the ACM result set.",
    ]
    (PACK_DIR / "logs/scan_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dirs()
    write_tables()
    write_figures()
    gray_count = write_gray_surfaces()
    write_snippets(gray_count)
    write_scan_summary()
    failed = PACK_DIR / "FAILED_ITEMS.md"
    if not failed.exists():
        failed.write_text("# Failed Items\n\nNo failed packaging items recorded so far.\n", encoding="utf-8")
    print(f"pack_dir={PACK_DIR}")
    print(f"gray_surface_cases={gray_count}")


if __name__ == "__main__":
    main()
