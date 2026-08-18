# Reproducibility contract

## Paper-specified settings

| Setting | Value |
|---|---:|
| Input | 800 x 800 |
| Detector | rotated FCOS, ResNet-50, FPN |
| Total iterations | 48,000 |
| Burn-in | 12,800 iterations |
| Batch size | 4 |
| Optimizer | Adam |
| Initial learning rate | 5e-5 |
| Weight decay | 0.05 |
| Gradient clip | max norm 35 |
| SCFE channels | 32, 64, 128 |
| InfoNCE temperature | 0.07 |
| SACC `(kernel,dilation)` | `(3,1)`, `(5,2)`, `(7,3)` |
| Table 9 parameters (baseline / +SWCS / +MAR / full) | 32.1M / 33.9M / 34.3M / 36.1M |
| PSKG Harris balance `zeta` | 0.04 |
| Keypoint relative threshold | 1% of map maximum |
| SALRP lambdas | target 1.05, background 0.95, hard 0.85 |
| Focal alpha / gamma | 0.25 / 2.0 |
| Hard-negative modulation | 0.4 |
| Center loss weight | 0.05 |
| Superpixel / overlap / edge weights | 5.0 / 10.0 / 0.3 |
| Reconstruction / distillation weights | 0.5 / 0.5 |
| Inference branch | student |

## Implementation defaults

The following implementation choices are exposed in code and configuration:

| Choice | Default | Origin/rationale |
|---|---:|---|
| Random seed | 42 | Fixed default; the paper does not report one. |
| SCFE pretraining | 20 epochs | Default contrastive-training schedule. |
| MoCo momentum / queue | 0.999 / 8192 | Contrastive encoder defaults. |
| EMA detector momentum | 0.9996 | Teacher-update default. |
| Pol-SLIC region / patch size | 24 / 96 | ACPC preprocessing defaults. |
| `tau_sal` / `tau_var` | 0.35 / 0.08 | Normalized Xpol scale; exposed as CLI options. |
| Uncertainty function `U` | Gaussian around response product 0.45, sigma 0.18 | The paper names `U` but gives no closed form. |
| GMM high / low posterior | 0.70 / 0.20 | Paper defines the sets but not cutoffs; exposed in config. |
| GMM high / uncertain confidence | 0.20 / 0.02 | Exposed in config. |
| Graph Top-K / KNN | 64 / 6 | Paper specifies Top-K/KNN but not K; exposed in config. |
| SACC descriptor reduction | 16 (`256/16=16` channels) | Eq. (17) does not state the descriptor width; this setting reproduces the +1.8M SWCS cost in Table 9. |
| ViT reconstructor depth | 4 layers | The text only says lightweight ViT; four layers reproduce the +2.2M MAR cost in Table 9. |
| EGCSF residual initialization | 0.1 | "Small value" in the paper; exposed in config. |
| Eq. (33) `thr` | 0.5 | Paper does not provide the numerical threshold; 0.5 makes "high-confidence negative" explicit and is configurable. |

Changing any entry in the second table creates a new implementation variant
and should be recorded with the experiment.

## Environment note

The paper reports PyTorch 2.3/CUDA 11.8, whereas the bundled training stack is
based on MMDetection 2.x/MMRotate 0.x. The method modules and CPU tests are
PyTorch-2.x compatible. End-to-end legacy CUDA training additionally requires
`mmcv-full` 1.7.1 compiled for the chosen PyTorch/CUDA pair. The included
legacy environment (PyTorch 1.13.1/CUDA 11.6/MMCV 1.7.1) is the tested
end-to-end path.
