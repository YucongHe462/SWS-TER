"""SAR-native boundary extraction used by the SWS-TER training path."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== Activation Function ========================
@torch.jit.script
def smish(input):
    """
    Applies the smish function element-wise:
    smish(x) = x * tanh(log(1 + sigmoid(x)))
    """
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))


class Smish(nn.Module):
    """Smish activation module."""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return smish(input)


# ======================== Weight Initialization ========================
def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# ======================== SAR Physics Front-End ========================
class SARStatisticsEncoder(nn.Module):
    """
    Physics-informed SAR preprocessing module.

    Transforms single-channel SAR intensity into a multi-channel representation
    encoding physical scattering properties:
      - Channel 0: Log-domain intensity (linearizes multiplicative noise)
      - Channel 1..N: Multi-scale Coefficient of Variation maps (CFAR edge metric)

    The CoV at scale k is computed as:
        CoV_k(x,y) = sqrt(Var_k[I]) / Mean_k[I]
    where Var_k and Mean_k are local statistics over a k x k window.

    Under H0 (homogeneous clutter): CoV = 1/sqrt(L), constant.
    Under H1 (scattering boundary): CoV >> 1/sqrt(L).

    Args:
        out_ch (int): Number of output channels. Default: 3.
        roa_windows (list[int]): Kernel sizes for multi-scale CoV computation.
            Default: [3, 5, 7].
    """
    def __init__(self, out_ch=3, roa_windows=[3, 5, 7]):
        super().__init__()
        self.roa_windows = roa_windows
        # Multi-scale CoV channels + 1 log-intensity channel
        in_ch = len(roa_windows) + 1
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            Smish()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) Single-channel SAR intensity image.

        Returns:
            (B, out_ch, H, W) Physics-encoded feature maps.
        """
        # Log-domain transform: linearizes multiplicative speckle
        # I = sigma0 * n  =>  log(I) = log(sigma0) + log(n)
        log_x = torch.log1p(x.clamp(min=0))
        channels = [log_x]

        for k in self.roa_windows:
            pad = k // 2
            # Local mean via average pooling
            avg = F.avg_pool2d(x, k, stride=1, padding=pad)
            # Local second moment
            avg_sq = F.avg_pool2d(x * x, k, stride=1, padding=pad)
            # Local variance = E[X^2] - E[X]^2
            local_var = (avg_sq - avg ** 2).clamp(min=1e-7)
            # Coefficient of Variation = sqrt(Var) / Mean
            # This is a sufficient statistic for edge detection under
            # multiplicative noise model (CFAR property)
            cov = local_var.sqrt() / (avg.abs() + 1e-7)
            channels.append(cov)

        # Learnable fusion: (B, in_ch, H, W) -> (B, out_ch, H, W)
        return self.fuse(torch.cat(channels, dim=1))


class BoundaryFusion(nn.Module):
    """Multi-scale edge map fusion via depthwise convolution + pixel shuffle."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3,
                                 stride=1, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24 * 1, kernel_size=3,
                                 stride=1, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return smish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True))
        self.add_module('smish1', Smish())
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(smish(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16
        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features=None,
                 stride=1, use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = Smish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class SARBoundaryExtractor(nn.Module):
    """Extract multi-scale SAR boundaries and an optional uncertainty map.

    A log-domain, multi-scale coefficient-of-variation encoder supplies the
    scattering statistics used by the compact boundary backbone. The
    uncertainty map indicates regions where boundary evidence is unreliable
    and allows downstream losses to discount heavy clutter.

    The module produces three scale-specific maps, one fused map, and—when
    requested—one uncertainty map.
    The uncertainty map indicates regions where edge detection is unreliable
    (e.g., heavy speckle, low SCR areas), enabling downstream EdgeLoss to
    adaptively discount these regions.

    Args:
        with_uncertainty (bool): Whether to output uncertainty map. Default: True.
        roa_windows (list[int]): Multi-scale CoV window sizes. Default: [3, 5, 7].
    """

    def __init__(self, with_uncertainty=True, roa_windows=[3, 5, 7]):
        super().__init__()

        # === SAR Physics-Informed Front-End ===
        self.statistics_encoder = SARStatisticsEncoder(
            out_ch=3, roa_windows=roa_windows)

        # Compact boundary backbone.
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Skip connections
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        # Upsampling blocks for multi-scale output
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        # Multi-scale fusion
        self.block_cat = BoundaryFusion(3, 3)

        # === Uncertainty Calibration Head ===
        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                Smish(),
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            )

        self.apply(weight_init)

    def _compute_cov_edge_prior(self, x):
        """
        Compute a physics-based edge prior directly from CoV statistics.

        This provides a meaningful edge map from iteration 0, without any
        learned parameters. It acts as a strong initialization that the
        backbone can refine during training.

        The CoV map is normalized to [0, 1] range and serves as a direct
        edge probability estimate under CFAR theory.

        Args:
            x: (B, 1, H, W) SAR intensity image (positive values).

        Returns:
            edge_prior: (B, 1, H, W) Normalized CoV-based edge map.
        """
        # Ensure positive values for CoV computation
        x_pos = x.clamp(min=1e-3)

        # Multi-scale CoV for robust edge estimation
        edge_maps = []
        for k in [5, 7]:
            pad = k // 2
            local_mean = F.avg_pool2d(x_pos, k, stride=1, padding=pad)
            local_sq_mean = F.avg_pool2d(x_pos ** 2, k, stride=1, padding=pad)
            local_var = (local_sq_mean - local_mean ** 2).clamp(min=1e-7)
            cov = local_var.sqrt() / (local_mean + 1e-7)
            edge_maps.append(cov)

        # Average multi-scale CoV maps
        edge_prior = sum(edge_maps) / len(edge_maps)

        # Normalize to [0, 1] per sample for stable edge magnitude
        B = edge_prior.shape[0]
        for i in range(B):
            e_min = edge_prior[i].min()
            e_max = edge_prior[i].max()
            if e_max - e_min > 1e-6:
                edge_prior[i] = (edge_prior[i] - e_min) / (e_max - e_min)
            else:
                edge_prior[i] = edge_prior[i] * 0

        return edge_prior

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) Single-channel SAR intensity image.
                (B, 3, H, W) is also accepted — will be converted to single-channel.

        Returns:
            results: List of tensors [out_1, out_2, out_3, fused_edge, uncertainty(optional)]
                - out_1, out_2, out_3: Multi-scale edge maps
                - fused_edge (results[3]): Final fused boundary field, (B, 1, H, W)
                - uncertainty (results[4], optional): Edge uncertainty map, (B, 1, H, W)
                  Values in [0, 1], higher means less reliable edge
        """
        assert x.ndim == 4, f"Expected 4D input, got shape {x.shape}"

        # Handle multi-channel input (e.g., if SAR image was duplicated to 3 channels)
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)  # Convert to single channel

        # Record original size for output alignment
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Pad input to be divisible by 8 (required by stride-2 backbone)
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # === Compute physics-based CoV edge prior (no learnable parameters) ===
        # This provides meaningful edge signal from iteration 0 (unlike random backbone).
        # CoV = sqrt(Var)/Mean is a CFAR edge detector for multiplicative noise.
        with torch.no_grad():
            cov_edge_prior = self._compute_cov_edge_prior(x)

        # Physics-informed preprocessing: (B,1,H,W) -> (B,3,H,W)
        x_feat = self.statistics_encoder(x)

        # Compact multi-scale boundary backbone.
        # Block 1
        block_1 = self.block_1(x_feat)               # [B, 16, H/2, W/2]
        block_1_side = self.side_1(block_1)     # [B, 32, H/4, W/4]

        # Block 2
        block_2 = self.block_2(block_1)         # [B, 32, H/2, W/2]
        block_2_down = self.maxpool(block_2)    # [B, 32, H/4, W/4]
        block_2_add = block_2_down + block_1_side

        # Block 3 (Dense)
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        # Multi-scale upsampling
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        results = [out_1, out_2, out_3]

        # Multi-scale fusion -> final boundary field
        block_cat = torch.cat(results, dim=1)   # [B, 3, H, W]
        block_cat = self.block_cat(block_cat)   # [B, 1, H, W]

        # === Residual fusion with CoV prior ===
        # The CoV prior ensures meaningful edge output even with untrained backbone.
        # As training progresses, the backbone learns to refine beyond the prior.
        # Interpolate prior to match backbone output size (may differ due to stride/padding)
        if cov_edge_prior.shape != block_cat.shape:
            cov_edge_prior = F.interpolate(
                cov_edge_prior, size=block_cat.shape[2:],
                mode='bilinear', align_corners=False)
        block_cat = block_cat + cov_edge_prior

        results.append(block_cat)

        # Uncertainty calibration head
        if self.with_uncertainty:
            uncertainty = self.uncertainty_head(block_cat.detach())
            results.append(uncertainty)

        # Crop back to original size if input was padded
        if pad_h > 0 or pad_w > 0:
            results = [r[:, :, :orig_h, :orig_w] for r in results]

        return results
