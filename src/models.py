"""U-Net variants for hologram -> complex field reconstruction.

Rather than three separately-defined classes (as in the original template),
depth/width are both controlled by `channels`, so "baseline / wide / deep" are
just different configs of the same architecture:

    channels=(64, 128)       -> baseline (2 encoder/decoder stages)
    channels=(128, 256)      -> wide     (2 stages, more channels per stage)
    channels=(64, 128, 256)  -> deep     (3 stages)
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, channels=(64, 128)):
        super().__init__()
        assert len(channels) >= 2, "need at least one down-stage + a bottleneck"

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels[:-1]:
            self.downs.append(DoubleConv(prev_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch
        self.bottleneck = DoubleConv(prev_ch, channels[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev_ch = channels[-1]
        for ch in reversed(channels[:-1]):
            self.ups.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch))  # *2: concatenated skip connection
            prev_ch = ch

        self.final_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.final_conv(x)  # channel 0: amplitude, channel 1: phase


UNET_CONFIGS = {
    "baseline": (64, 128),
    "wide": (128, 256),
    "deep": (64, 128, 256),
}
