"""
Pinwheel-shaped Convolution (PConv) module for infrared small target detection.

Reference: JN-Yang/PConv-SDloss-Data
Paper: PConv — 一种用于红外小目标检测的扇形卷积
"""

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BN + SiLU — same as ultralytics Conv."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PConv(nn.Module):
    """
    Pinwheel-shaped Convolution using Asymmetric Padding.

    Replaces standard Conv with a pinwheel-shaped receptive field that better captures
    small infrared targets by emphasizing cross-shaped spatial features.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (typically 3)
        s (int): Stride (typically 2 for downsampling layers)
    """

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        # Four asymmetric padding directions: (left, right, top, bottom)
        # Each creates a directional pad that feeds into a specific branch
        padding_configs = [
            (k, 0, 1, 0),  # pad left side
            (0, k, 0, 1),  # pad right side
            (0, 1, k, 0),  # pad top
            (1, 0, 0, k),  # pad bottom
        ]
        self.pad = nn.ModuleList([nn.ZeroPad2d(padding=p) for p in padding_configs])

        # Horizontal kernel branches: (1, k) — captures horizontal strips
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        # Vertical kernel branches: (k, 1) — captures vertical strips
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # Fusion conv to combine the four directional branches
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # Four directional branches
        yw0 = self.cw(self.pad[0](x))  # left-padded → horizontal kernel
        yw1 = self.cw(self.pad[1](x))  # right-padded → horizontal kernel
        yh0 = self.ch(self.pad[2](x))  # top-padded → vertical kernel
        yh1 = self.ch(self.pad[3](x))  # bottom-padded → vertical kernel

        # Concatenate and fuse
        out = torch.cat([yw0, yw1, yh0, yh1], dim=1)
        return self.cat(out)
