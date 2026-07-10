import sys

import torch


sys.path.insert(0, r"C:\Users\bbqsa\Desktop\红外检测\infrared_detection_code (1)\acm")
from model.metrics import SamplewiseSigmoidMetric


def run_case(name, pred_mask, label_mask):
    metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    output = torch.full((1, 1, 4, 4), -10.0)
    label = torch.zeros((1, 1, 4, 4))
    for y, x in pred_mask:
        output[0, 0, y, x] = 10.0
    for y, x in label_mask:
        label[0, 0, y, x] = 1.0
    metric.update(output, label)
    ious, miou = metric.get()
    print(
        name,
        "IoU_array=",
        ious,
        "mIoU=",
        float(miou),
        "inter=",
        metric.total_inter,
        "union=",
        metric.total_union,
    )
    return float(miou)


def main():
    full = run_case("full_overlap", [(1, 1), (1, 2)], [(1, 1), (1, 2)])
    none = run_case("no_overlap", [(0, 0)], [(3, 3)])
    partial = run_case("partial_overlap", [(0, 0), (0, 1)], [(0, 1), (0, 2)])
    empty = run_case("both_empty", [], [])

    assert abs(full - 1.0) < 1e-9, full
    assert abs(none - 0.0) < 1e-9, none
    assert 0.0 < partial < 1.0, partial
    assert abs(partial - (1.0 / 3.0)) < 1e-9, partial
    assert abs(empty - 0.0) < 1e-9, empty
    print("metric_unit_tests_ok")
    print("empty_empty_convention=0_with_existing_epsilon_denominator")


if __name__ == "__main__":
    main()
