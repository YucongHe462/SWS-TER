import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePopulationPerceptionBranch(nn.Module):
    """Lightweight region-prior branch for sparse partial weak supervision.

    This branch does not produce detection boxes. It estimates multi-level
    targetness and soft priors, which are later converted to modulation maps for
    the detection head.
    """

    def __init__(self, in_channels, feat_channels=64, num_levels=5):
        super().__init__()
        self.num_levels = num_levels
        self.level_encoders = nn.ModuleList()
        self.targetness_heads = nn.ModuleList()
        self.scale_heads = nn.ModuleList()
        self.uncertainty_heads = nn.ModuleList()

        for _ in range(num_levels):
            self.level_encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feat_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            self.targetness_heads.append(nn.Conv2d(feat_channels, 1, 1))
            self.scale_heads.append(nn.Conv2d(feat_channels, 1, 1))
            self.uncertainty_heads.append(nn.Conv2d(feat_channels, 1, 1))

    def forward(self, feats):
        encoded_feats = []
        targetness_maps = []
        scale_maps = []
        uncertainty_maps = []

        for feat, encoder, targetness_head, scale_head, unc_head in zip(
                feats, self.level_encoders, self.targetness_heads,
                self.scale_heads, self.uncertainty_heads):
            encoded = encoder(feat)
            encoded_feats.append(encoded)
            targetness_maps.append(torch.sigmoid(targetness_head(encoded)))
            scale_maps.append(torch.sigmoid(scale_head(encoded)))
            uncertainty_maps.append(torch.sigmoid(unc_head(encoded)))

        return dict(
            encoded_feats=encoded_feats,
            M_g=targetness_maps,
            R_soft=targetness_maps,
            Z_scale=scale_maps,
            Z_unc=uncertainty_maps,
        )


class GlobalLocalPriorAdapter(nn.Module):
    """Convert SPP priors to soft residual modulation signals.

    The adapter only modulates features. It does not hard filter positions and
    does not suppress low-response areas to zero.
    """

    def __init__(self,
                 in_channels,
                 prior_channels=64,
                 num_levels=5,
                 alpha=0.1,
                 gamma_min=0.5,
                 gamma_max=1.5):
        super().__init__()
        self.num_levels = num_levels
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.cls_generators = nn.ModuleList()
        self.reg_generators = nn.ModuleList()
        self.assign_generators = nn.ModuleList()
        self.feature_convs = nn.ModuleList()

        for _ in range(num_levels):
            self.cls_generators.append(
                nn.Sequential(
                    nn.Conv2d(prior_channels + 3, prior_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(prior_channels, 1, 1),
                )
            )
            self.reg_generators.append(
                nn.Sequential(
                    nn.Conv2d(prior_channels + 3, prior_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(prior_channels, 1, 1),
                )
            )
            self.assign_generators.append(
                nn.Sequential(
                    nn.Conv2d(prior_channels + 3, prior_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(prior_channels, 1, 1),
                )
            )
            self.feature_convs.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))

    def forward(self, feats, spp_outputs):
        modulated_feats = []
        g_cls = []
        g_reg = []
        g_assign = []

        for feat, prior_feat, targetness, scale, uncertainty, cls_gen, reg_gen, assign_gen, feat_conv in zip(
                feats,
                spp_outputs['encoded_feats'],
                spp_outputs['M_g'],
                spp_outputs['Z_scale'],
                spp_outputs['Z_unc'],
                self.cls_generators,
                self.reg_generators,
                self.assign_generators,
                self.feature_convs):
            prior_input = torch.cat([prior_feat, targetness, scale, uncertainty], dim=1)
            cls_gate = torch.sigmoid(cls_gen(prior_input))
            reg_gate = torch.sigmoid(reg_gen(prior_input))
            assign_gate = torch.sigmoid(assign_gen(prior_input))
            fused_gate = 0.5 * (cls_gate + reg_gate)
            modulated = feat + self.alpha * fused_gate * feat_conv(feat)
            modulated_feats.append(modulated)
            g_cls.append(cls_gate)
            g_reg.append(reg_gate)
            g_assign.append(assign_gate)

        spp_outputs = dict(spp_outputs)
        spp_outputs.update(G_cls=g_cls, G_reg=g_reg, G_assign=g_assign)
        return modulated_feats, spp_outputs


def build_spp_prior_modules(cfg, in_channels, num_levels):
    cfg = dict(cfg or {})
    feat_channels = cfg.get('feat_channels', 64)
    alpha = cfg.get('alpha', 0.1)
    branch = SparsePopulationPerceptionBranch(
        in_channels=in_channels,
        feat_channels=feat_channels,
        num_levels=num_levels)
    adapter = GlobalLocalPriorAdapter(
        in_channels=in_channels,
        prior_channels=feat_channels,
        num_levels=num_levels,
        alpha=alpha,
        gamma_min=cfg.get('gamma_min', 0.5),
        gamma_max=cfg.get('gamma_max', 1.5))
    return branch, adapter
