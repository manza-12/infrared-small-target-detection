# nIoU Metric Fix Note

## Original Error

- `TypeError: only 0-dimensional arrays can be converted to Python scalars`
- `ValueError: setting an array element with a sequence`
- Location: `acm/model/metrics.py` line 110

## Cause

- `np.histogram` returned an array with `shape=(1,)`.
- `area_inter_arr[b]` is a scalar slot.
- The current task is binary infrared small target segmentation, so histogram-based multi-class statistics are not suitable for this nIoU path.

## Change

- Apply sigmoid and threshold to `output`.
- Binarize `label` with `label > 0`.
- Squeeze each sample to `H x W`.
- Use `logical_and` / `logical_or` to count per-image intersection and union.

## Unit Tests

- `full_overlap`: IoU = 1.0
- `no_overlap`: IoU = 0.0
- `partial_overlap`: IoU = 0.3333333333333333
- `both_empty`: IoU = 0.0

## Person1 1 Epoch Validation

- `FPN + BiLocal`: passed
- `FPN + AsymBi`: passed
