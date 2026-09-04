"""SGC-Net (SwiGLU-Gated Crack Network).

A shallow 3-level encoder-decoder (U-Net style) with a SwiGLU neck on each
level and skip connections that fuse the encoder features back into the
decoder, yielding a single full-resolution logit map.  Tuned for thin (1-3 px)
cracks at 8x downsample.

Design notes
------------
* conv_t = 64      : plain conv for the shallow level, depthwise-separable
                     conv on the deeper levels (parameter efficiency).
* _ch_ks           : depthwise kernel size per level. A DW conv has rank k^2,
                     so it is kept >= channels/3 to stay well-conditioned for
                     the Muon optimizer.
* No Transformer / SPPF / PAN -- thin cracks need local precision, not global
  context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Global hyper-parameters                                                     #
# --------------------------------------------------------------------------- #
dropout = 0.015
dropout_t = 32

# Channels at/above which a depthwise-separable conv is used (see
# DepthwiseSeparableConv).  Kernel sizes per level come from _ch_ks so that the
# DW rank k^2 stays Muon-friendly.
conv_t = 64
_ch_ks = {256: 7, 128: 7, 64: 5}


def group_norm_channel(num_channels: int, num_groups: int = 8, min_channels: int = 4) -> int:
    """Return the largest group count that divides num_channels (> min_channels)."""
    if num_channels < min_channels:
        return 1
    num_groups = min(num_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    if num_channels // num_groups < min_channels:
        num_groups -= 1
        while num_channels % num_groups != 0:
            num_groups -= 1
    return num_groups


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv + GN + SiLU followed by a 1x1 pointwise projection."""

    def __init__(self, inputNum: int, outputNum: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, act: bool = True) -> None:
        super().__init__()
        if act:
            self.conv = nn.Sequential(
                nn.Conv2d(inputNum, inputNum, kernel_size, stride=stride,
                          dilation=dilation, padding=kernel_size // 2 + dilation - 1,
                          padding_mode='reflect', groups=inputNum, bias=False),
                nn.GroupNorm(group_norm_channel(inputNum), inputNum),
                nn.SiLU(),
                nn.Conv2d(inputNum, outputNum, 1, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inputNum, inputNum, kernel_size, stride=stride,
                          dilation=dilation, padding=kernel_size // 2 + dilation - 1,
                          padding_mode='reflect', groups=inputNum, bias=False),
                nn.GroupNorm(group_norm_channel(inputNum), inputNum),
                nn.Conv2d(inputNum, outputNum, 1, bias=False),
            )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class conv(nn.Module):
    """Conv + GN + SiLU. Uses a DW conv when both channels >= conv_t (64)."""

    def __init__(self, inputNum: int, outputNum: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, act: bool = True) -> None:
        super().__init__()
        if outputNum >= conv_t and inputNum >= conv_t and kernel_size != 1:
            self.conv = DepthwiseSeparableConv(inputNum, outputNum,
                                               _ch_ks.get(inputNum, 3), stride, dilation, act)
        else:
            self.conv = nn.Conv2d(inputNum, outputNum, kernel_size,
                                  stride=stride, dilation=dilation,
                                  padding=kernel_size // 2 + dilation - 1,
                                  padding_mode='reflect')
        self.norm = nn.GroupNorm(group_norm_channel(outputNum), outputNum)
        self.act = act

    def forward(self, x: torch.Tensor):
        if self.act:
            return F.silu(self.norm(self.conv(x)))
        else:
            return self.norm(self.conv(x))


class DepthwiseSeparableConvT(nn.Module):
    """Transposed counterpart of DepthwiseSeparableConv (used for upsampling)."""

    def __init__(self, inputNum: int, outputNum: int, kernel_size: int = 3,
                 stride: int = 1) -> None:
        super().__init__()
        self.convT = nn.Sequential(
            nn.Conv2d(inputNum, outputNum, 1, bias=False),
            nn.GroupNorm(group_norm_channel(outputNum), outputNum),
            nn.ConvTranspose2d(outputNum, outputNum, kernel_size, stride=stride,
                               padding=kernel_size // 2, output_padding=stride - 1,
                               groups=outputNum),
        )

    def forward(self, x: torch.Tensor):
        return self.convT(x)


class convT(nn.Module):
    """ConvTranspose + GN + SiLU. Uses a DW transpose when both channels >= conv_t."""

    def __init__(self, inputNum: int, outputNum: int, kernel_size: int = 3,
                 stride: int = 1) -> None:
        super().__init__()
        if outputNum >= conv_t and inputNum >= conv_t:
            self.convT = DepthwiseSeparableConvT(inputNum, outputNum,
                                                 _ch_ks.get(outputNum, 3), stride)
        else:
            self.convT = nn.ConvTranspose2d(inputNum, outputNum, kernel_size,
                                            stride=stride, padding=kernel_size // 2,
                                            output_padding=stride - 1)
        self.norm = nn.GroupNorm(group_norm_channel(outputNum), outputNum)

    def forward(self, x: torch.Tensor):
        return F.silu(self.norm(self.convT(x)))


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward block with a residual connection.

    gate and up are 3x3 convs, fused by a 1x1 projection after
    SiLU(gate) * up.  Used as the bottleneck neck on every scale.
    """

    def __init__(self, dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim == 0:
            hidden_dim = dim * 8 // 3
        self.gate_proj = conv(dim, hidden_dim, 3, act=False)
        self.up_proj = conv(dim, hidden_dim, 3, act=False)
        self.down_proj = conv(hidden_dim, dim, 1, act=False)
        if dim >= dropout_t and dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))) + x


class downsample(nn.Module):
    """2x downsampling: strided conv plus an avg-pool residual branch."""

    def __init__(self, inputNum: int, outputNum: int) -> None:
        super().__init__()
        self.conv = conv(inputNum, outputNum, 3, 2)
        self.res = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(inputNum, outputNum, 1, bias=False),
            nn.GroupNorm(group_norm_channel(outputNum), outputNum),
            nn.SiLU(),
        )
        self.norm = nn.GroupNorm(group_norm_channel(outputNum), outputNum)
        if outputNum >= dropout_t and dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.drop(self.norm(self.conv(x) + self.res(x)))


class upsample(nn.Module):
    """2x upsampling: transposed conv plus a bilinear-upsample residual branch."""

    def __init__(self, inputNum: int, outputNum: int) -> None:
        super().__init__()
        self.up = convT(inputNum, outputNum, 3, 2)
        self.res = nn.Sequential(
            nn.Conv2d(inputNum, outputNum, 1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.GroupNorm(group_norm_channel(outputNum), outputNum),
            nn.SiLU(),
        )
        self.norm = nn.GroupNorm(group_norm_channel(outputNum), outputNum)
        if outputNum >= dropout_t and dropout > 0:
            self.drop = nn.Dropout2d(dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.drop(self.norm(self.up(x) + self.res(x)))


class SGCNet(nn.Module):
    """SGC-Net: 3-level encoder-decoder with skip connections and SwiGLU-gated necks.

    Encoder: downsample -> SwiGLU neck (residual).  Decoder: upsample + SwiGLU
    neck that fuses each encoder feature (skip connection) through a
    concatenation + conv, then a final 1x1 projection to output_channel logits.
    """

    def __init__(
        self,
        DownNum: int = 3,
        DownNeckNum: int | list[int] = [3, 2, 1],
        channel_sizes: list[int] = [16, 64, 128],
        UpNeckNum: int | list[int] = [0, 2, 2],
        input_channel: int = 3,
        output_channel: int = 1,
    ) -> None:
        super().__init__()

        if len(channel_sizes) != DownNum:
            raise ValueError("channel_sizes length must be DownNum")
        if isinstance(DownNeckNum, list) and len(DownNeckNum) != DownNum:
            raise ValueError("DownNeckNum list must be DownNum")
        if isinstance(UpNeckNum, list) and len(UpNeckNum) != DownNum:
            raise ValueError("UpNeckNum list must be DownNum")

        channel_sizes = [input_channel] + channel_sizes

        # --- Encoder: downsample + SwiGLU neck on each scale -------------------
        self.downs = nn.ModuleList()
        for i in range(DownNum):
            self.downs.append(downsample(channel_sizes[i], channel_sizes[i + 1]))

        def _build_neck(num_blocks, C):
            neck = nn.ModuleList()
            for _ in range(num_blocks):
                neck.append(SwiGLU(C))
            return neck

        self.downNecks = nn.ModuleList()
        for i in range(DownNum):
            n = DownNeckNum[i] if isinstance(DownNeckNum, list) else DownNeckNum
            self.downNecks.append(_build_neck(n, channel_sizes[i + 1]))

        channel_sizes = [channel_sizes[1]] + channel_sizes[1:]

        # --- Decoder: upsample + neck + skip-connection fusion -----------------
        self.ups = nn.ModuleList()
        self.fus_convs = nn.ModuleList()
        self.upNecks = nn.ModuleList()

        for i in range(DownNum):
            C_cur = channel_sizes[-(i + 1)]
            self.ups.append(upsample(C_cur, channel_sizes[-(i + 2)]))
            n = UpNeckNum[i] if isinstance(UpNeckNum, list) else UpNeckNum
            self.upNecks.append(_build_neck(n, C_cur))
            # DW 3x3 spatial mixing + 1x1 channel projection
            k_size = _ch_ks.get(C_cur * 2, 3)
            self.fus_convs.append(nn.Sequential(
                nn.Conv2d(C_cur * 2, C_cur * 2, k_size, padding=k_size // 2, padding_mode='reflect',
                          groups=C_cur * 2, bias=False),
                nn.Conv2d(C_cur * 2, C_cur, 1, bias=False),
            ))

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.output_conv = nn.Sequential(
            conv(channel_sizes[0], channel_sizes[0], 3),
            nn.Conv2d(channel_sizes[0], output_channel, 1, bias=False),
        )

    def _neck_way(self, x: torch.Tensor, neck: nn.ModuleList) -> torch.Tensor:
        """Run a stack of neck blocks (SwiGLU) sequentially."""
        for block in neck:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder ----------------------------------------------------------- #
        xinputs = []
        for down, down_neck in zip(self.downs, self.downNecks):
            x = down(x)
            x = self._neck_way(x, down_neck) + x
            xinputs.append(x)

        # --- Decoder (with skip connections) ------------------------------------ #
        for up, up_neck, fus_conv in zip(self.ups, self.upNecks, self.fus_convs):
            x_res = xinputs.pop()
            x = self._neck_way(x, up_neck) + x
            x = fus_conv(torch.cat([x, x_res], dim=1))
            x = up(x)

        # --- Output logits ----------------------------------------------------- #
        x = self.output_conv(x)
        return x


# ======================= Test =======================

if __name__ == '__main__':
    import torchinfo
    model = SGCNet().eval()
    y = model(torch.randn(1, 3, 512, 512))
    print(f"Output shape: {list(y.shape)}")
    s = torchinfo.summary(model, input_size=(1, 3, 512, 512), mode='eval',
                          col_names=["output_size", "num_params"], depth=1, verbose=0)
    print(f"\nTotal params: {s.total_params:,} / 490,000")
    print(f"Headroom: {490_000 - s.total_params:,}")
