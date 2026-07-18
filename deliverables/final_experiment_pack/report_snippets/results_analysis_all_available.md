# Results Analysis for All Available ACM Models

## Final Six-Model Main Table

| Person | Backbone | Fusion | IoU | nIoU | Pd | Fa |
|---|---|---|---:|---:|---:|---:|
| person1 | FPN | BiLocal | 0.2664489244 | 0.3093132385 | 0.9875000 | 704.678125 |
| person1 | FPN | AsymBi | 0.4059420923 | 0.4855873167 | 0.9906250 | 360.070312 |
| person2 | FPN | BiGlobal | 0.3660601796 | 0.4457140235 | 0.9890625 | 440.821875 |
| person2 | UNet | BiLocal | 0.3772599337 | 0.4348144315 | 0.9859375 | 415.790625 |
| person3 | UNet | AsymBi | 0.3262435852 | 0.3772964333 | 0.9953125 | 530.640625 |
| person4 | UNet | BiGlobal | 0.3883507577 | 0.4837368390 | 0.9937500 | 389.137500 |


## FPN Branch

For FPN, the best IoU is achieved by FPN+AsymBi (0.4059), followed by FPN+BiGlobal (0.3661) and FPN+BiLocal (0.2664). The same ordering is also observed for nIoU: AsymBi (0.4856) > BiGlobal (0.4457) > BiLocal (0.3093). In this branch, AsymBi is the strongest fusion strategy, while BiLocal is the weakest on this fixed split.

## UNet Branch

For UNet, the best IoU is achieved by UNet+BiGlobal (0.3884), followed by UNet+BiLocal (0.3773) and UNet+AsymBi (0.3262). For nIoU, UNet+BiGlobal is also best (0.4837), followed by UNet+BiLocal (0.4348) and UNet+AsymBi (0.3773).

## Backbone Comparison

Under BiLocal, UNet+BiLocal improves IoU over FPN+BiLocal, but FPN+BiLocal has slightly higher Pd. Under AsymBi, FPN+AsymBi is better than UNet+AsymBi in IoU and nIoU, while UNet+AsymBi has the highest Pd among all models. Under BiGlobal, UNet+BiGlobal has higher IoU and nIoU than FPN+BiGlobal and also lower Fa.

## Overall Observations

- Highest IoU: FPN+AsymBi (0.4059)
- Highest nIoU: FPN+AsymBi (0.4856)
- Highest Pd: UNet+AsymBi (0.9953)
- Lowest Fa: FPN+AsymBi (360.07)

The metrics show trade-offs. FPN+AsymBi has the strongest IoU/nIoU in the FPN branch, UNet+BiGlobal is strongest in the UNet branch, and UNet+AsymBi has the highest Pd but not the best IoU. Therefore, no model should be described as universally best based on a single metric alone.
