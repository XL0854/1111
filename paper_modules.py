"""Optional paper-inspired alignment and seam-fusion adapters.

All newly introduced research components live in this file so the original
fixed-half RopStitch + PixelStitch pipeline remains easy to audit.  The
baseline only imports and calls the public helpers defined below.
"""

import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleDualAttention(nn.Module):
    """Residual multi-scale spatial/channel attention for a 2-channel CCL map.

    The residual gain starts at zero.  Therefore an old alignment checkpoint
    initially reproduces the baseline exactly and the adapter is learned
    gradually instead of perturbing the correlation field at load time.
    """

    def __init__(self, channels=2, width=12):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, width, 3, padding=dilation,
                          dilation=dilation, bias=False),
                nn.GroupNorm(3, width),
                nn.GELU(),
            )
            for dilation in (1, 2, 3)
        ])
        self.project = nn.Conv2d(width * 3, channels, 1, bias=False)
        hidden = max(2, channels)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, correlation):
        context = self.project(torch.cat([branch(correlation) for branch in self.branches], 1))
        channel = self.channel_attention(context)
        spatial = self.spatial_attention(torch.cat(
            (context.mean(1, keepdim=True), context.amax(1, keepdim=True)), 1
        ))
        return correlation + torch.tanh(self.gain) * context * channel * spatial


class LocalGlobalBlock(nn.Module):
    """CNN local path plus dilated context path with channel attention."""

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 1),
            nn.Sigmoid(),
        )
        self.output = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, inputs):
        paths = torch.cat((self.local(inputs), self.context(inputs)), 1)
        return inputs + self.output(paths * self.mix(paths))


class DifferenceAwareSeamFusion(nn.Module):
    """Predict one soft seam mask from two already aligned image views."""

    def __init__(self, width=24):
        super().__init__()
        self.shared_detail = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.GELU(),
        )
        self.input = nn.Conv2d(width * 3 + 2, width, 1)
        self.body = nn.Sequential(LocalGlobalBlock(width), LocalGlobalBlock(width))
        self.head = nn.Sequential(nn.Conv2d(width, 1, 1), nn.Sigmoid())

    def forward(self, images, support):
        if images.shape[0] != 2 or support.shape[0] != 2:
            raise ValueError("Seam fusion expects exactly two aligned views")
        valid = support.clamp(0, 1)
        left = self.shared_detail(images[0:1] * valid[0:1])
        right = self.shared_detail(images[1:2] * valid[1:2])
        features = torch.cat((left, right, (left - right).abs(),
                              valid[0:1], valid[1:2]), 1)
        probability = self.head(self.body(self.input(features)))
        weights = torch.cat((probability * valid[0:1],
                             (1 - probability) * valid[1:2]), 0)
        weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
        fused = (images * weights).sum(0, keepdim=True)
        union = valid.amax(0, keepdim=True)
        return fused * union, probability, weights


def _masked_mean(values, mask):
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


def frequency_photo(images, mask):
    """Low-frequency structure and high-frequency edge agreement losses."""
    low = F.avg_pool2d(images, 5, stride=1, padding=2)
    low_loss = _masked_mean((low[0:1] - low[1:2]).abs(), mask)
    dx = images[:, :, :, 1:] - images[:, :, :, :-1]
    dy = images[:, :, 1:, :] - images[:, :, :-1, :]
    edge_x = _masked_mean((dx[0:1] - dx[1:2]).abs(), mask[:, :, :, 1:] * mask[:, :, :, :-1])
    edge_y = _masked_mean((dy[0:1] - dy[1:2]).abs(), mask[:, :, 1:, :] * mask[:, :, :-1, :])
    return low_loss, 0.5 * (edge_x + edge_y)


def frequency_flow_objective(canvas, residual, cycle, base, base_objective, weights):
    """Add frequency supervision without modifying the original objective."""
    total, terms = base_objective(canvas, residual, cycle, base, weights["baseline"])
    from model import fixed_overlap, refine
    images, _ = refine(canvas, base, residual)
    low, high = frequency_photo(images, fixed_overlap(canvas))
    terms.update(freq_low=low, freq_high=high)
    total = total + weights["freq_low"] * low + weights["freq_high"] * high
    return total, terms


def _inner_boundary(mask, radius=2):
    outside = 1 - mask
    return mask * F.max_pool2d(outside, radius * 2 + 1, stride=1, padding=radius)


def seam_objective(images, support, fused, probability):
    """Normalized boundary anchoring and discrepancy-aware seam regularity."""
    valid = (support > 0.99).float()
    boundaries = torch.cat((_inner_boundary(valid[0:1]), _inner_boundary(valid[1:2])), 0)
    anchor = _masked_mean((fused - images[0:1]).abs(), boundaries[0:1])
    anchor = anchor + _masked_mean((fused - images[1:2]).abs(), boundaries[1:2])

    coarse = F.avg_pool2d(images, 5, stride=1, padding=2)
    discrepancy = (coarse[0:1] - coarse[1:2]).square().mean(1, keepdim=True).sqrt().detach()
    overlap = valid.amin(0, keepdim=True)
    px = (probability[:, :, :, 1:] - probability[:, :, :, :-1]).abs()
    py = (probability[:, :, 1:, :] - probability[:, :, :-1, :]).abs()
    wx = 0.05 + 0.5 * (discrepancy[:, :, :, 1:] + discrepancy[:, :, :, :-1])
    wy = 0.05 + 0.5 * (discrepancy[:, :, 1:, :] + discrepancy[:, :, :-1, :])
    sem = _masked_mean(px * wx, overlap[:, :, :, 1:] * overlap[:, :, :, :-1])
    sem = sem + _masked_mean(py * wy, overlap[:, :, 1:, :] * overlap[:, :, :-1, :])

    fx = (fused[:, :, :, 1:] - fused[:, :, :, :-1]).abs().mean(1, keepdim=True)
    fy = (fused[:, :, 1:, :] - fused[:, :, :-1, :]).abs().mean(1, keepdim=True)
    visual = _masked_mean(px * fx, overlap[:, :, :, 1:] * overlap[:, :, :, :-1])
    visual = visual + _masked_mean(py * fy, overlap[:, :, 1:, :] * overlap[:, :, :-1, :])
    terms = dict(seam_anchor=anchor, seam_semantic=sem, seam_visual=visual)
    return anchor + 0.2 * sem + 0.1 * visual, terms


def attach_correlation_adapter(geometry, adapter):
    """Patch only this model instance; the original comparison arm is untouched."""
    original = geometry.CCL

    def adapted(feature_1, feature_2):
        return adapter(original(feature_1, feature_2))

    geometry.CCL = adapted
