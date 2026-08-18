<h1 align="center">SWS-TER</h1>

<p align="center">
  <strong>Sparse Weakly Semi-supervised Tri-Evidence Recovery Network<br>
  for Oriented Ship Detection in PolSAR Imagery</strong>
</p>

<p align="center">
  Yucong He · Gui Gao · Tianwen Zhang · Dunyun He
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9">
  <img src="https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 1.13.1">
  <img src="https://img.shields.io/badge/CUDA-11.6-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 11.6">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square" alt="Apache-2.0 License"></a>
</p>

<p align="center">
  <a href="https://drive.google.com/drive/folders/10dqewpIj7hA5NvQ7XihiB9I73-Kkzptb?usp=sharing">
    <img src="https://img.shields.io/badge/Dataset-Shared_Folder-2E8B57?style=for-the-badge&logo=googledrive&logoColor=white" alt="Download the shared SWS-TER dataset">
  </a>
</p>

<p align="center">
  Official PyTorch implementation of the SWS-TER manuscript.
</p>

---

## 🚀 Overview

SWS-TER targets sparse, mixed-form supervision for oriented ship detection in
PolSAR imagery. It is built on MMDetection and MMRotate and implements the
complete three-stage training path described in the manuscript.

<p align="center">
  <img src="resources/sws_ter_architecture.jpg" width="100%" alt="SWS-TER network architecture">
</p>

<p align="center"><em>Overall architecture of the SWS-TER framework.</em></p>

### Main components

- **ACPC (Stage I):** polarimetric superpixels, revised Wishart saliency,
  region/background separability, four region anchors, diversity sampling,
  momentum SCFE, InfoNCE queue, prototype similarities and continuous
  `P_tar/P_bg/P_hard` priors
- **SWCS (Stages II-III):** scale-adaptive context compensation (SACC),
  prior-guided scattering keypoint graph (PSKG), GraphSAGE reasoning and
  evidence-guided context-scattering fusion (EGCSF)
- **SALRP:** ACPC-modulated sparse focal loss using Eq. (32), assigned by exact
  superpixel ownership and applied only to high-confidence negative FCOS
  locations in Eq. (33)
- **UGSRT:** per-FPN-level two-component GMM, teacher-student consistency,
  high/uncertain/low candidate partitioning, latent-feature mask tokens,
  online class prototypes and a lightweight ViT reconstructor
- **Mixed weak annotations:** RBox, HBox and Point supervision with center,
  superpixel/Voronoi, overlap and SAR edge-alignment constraints

The custom modules live under `projects/SWS_TER`, while the bundled legacy
`mmdet`, `mmrotate` and `semi_mmrotate` packages provide the end-to-end
training path.

---

## 🛠️ Installation

The tested compatibility path is Python 3.9, PyTorch 1.13.1, CUDA 11.6 and
`mmcv-full` 1.7.1:

```bash
git clone https://github.com/YucongHe462/SWS-TER.git
cd SWS-TER
conda env create -f requirements/environment-legacy.yml
conda activate sws-ter-legacy
```

`requirements/paper.txt` records the environment used for the paper. See
`docs/REPRODUCIBILITY.md` for compatibility notes and fixed experiment
settings.

---

## 📦 Dataset

The dataset used in this work is publicly shared for benchmarking and
reproducibility.

**Shared download link:**
[Open the SWS-TER dataset shared folder](https://drive.google.com/drive/folders/10dqewpIj7hA5NvQ7XihiB9I73-Kkzptb?usp=sharing)

> 📌 **Access note:** The hyperlink above is the complete shared-folder URL,
> including the `usp=sharing` parameter. Dataset files are not stored in this
> Git repository.

---

## ⚙️ Data preparation and Stage I priors

The network accepts ordinary single-channel SAR tiles. They remain
single-channel for SAR evidence computation and are replicated only at the
ImageNet-pretrained ResNet stem.

Set `SWS_TER_DATA_ROOT` to the directory containing `semi_ratio_20`,
`test_image` and `test_annotation`, then verify the released split:

```powershell
$env:SWS_TER_DATA_ROOT = 'D:\datasets\ospan'
python tools\verify_ospan_data.py `
  --data-root $env:SWS_TER_DATA_ROOT `
  --expect-counts
```

To rebuild the split from Pascal VOC XML files instead:

```bash
python tools/prepare_gr_dataset.py \
  --images data/raw/sar_gray \
  --annotations data/raw/Annotations \
  --out-dir data/ospan_sws_ter \
  --seed 42
```

Run the annotation-free ACPC stage on all training images:

```powershell
$TrainSplit = Join-Path $env:SWS_TER_DATA_ROOT `
  'semi_ratio_20\sparse_ratio_20'

python tools\acpc\run_acpc.py `
  --xpol-dirs (Join-Path $TrainSplit 'label_image') `
               (Join-Path $TrainSplit 'unlabel_image') `
  --work-dir work_dirs\acpc `
  --epochs 20 `
  --device cuda
```

The final `work_dirs/acpc/acpc_priors.json` file is consumed by the student
head and PSKG support map. Optional polarimetric preparation utilities are
retained for ablation studies but are not required by the single-channel path.

---

## 🚦 Training and evaluation

The released OSPAN 20%-image / 20%-instance HBox configuration is:

```bash
python tools/train.py configs/sws_ter/sws_ter_ospan_hbox_20_20.py \
  --work-dir work_dirs/sws_ter_ospan_hbox_20_20

python tools/test.py configs/sws_ter/sws_ter_ospan_hbox_20_20.py \
  work_dirs/sws_ter_ospan_hbox_20_20/latest.pth \
  --eval mAP
```

The published schedule is encoded in the config: 48,000 iterations, batch
size 4, 12,800 burn-in iterations, 800 x 800 inputs, Adam at `5e-5`, weight
decay `0.05`, linear warm-up and gradient clipping at norm 35. Inference uses
the student branch.

For separate Stage II/Stage III commands and a short end-to-end runtime check,
see `docs/RUN_OSPAN.md`.

---

## ✅ Verification

Core components have CPU-only unit tests and do not require MMCV CUDA ops:

```bash
pytest -q tests/test_core_modules.py
python -m compileall projects tools configs
```

The framework-level check requires the compiled legacy `mmcv-full`, but does
not require dataset files or a checkpoint:

```bash
python tests/framework_smoke.py
```

It builds the full model and checks the single-channel forward path, UGSRT,
Eq. (32) superpixel mapping, Eq. (33) gating, center-loss weight, student
inference and model parameter counts.

---

## 📚 Documentation

- `docs/METHOD_TO_CODE.md` — equation-by-equation implementation map
- `docs/REPRODUCIBILITY.md` — settings, seeds and implementation choices
- `docs/DATA.md` — directory layout and annotation encoding
- `docs/RUN_OSPAN.md` — complete PowerShell commands for every stage

---

## 📌 Citation

If you find this repository useful in your research, please cite the SWS-TER
manuscript. Machine-readable citation metadata is provided in `CITATION.cff`.

---

## 📄 License and attribution

SWS-TER is released under the Apache License 2.0. Bundled OpenMMLab-derived
code retains its original Apache-2.0 notices under `licenses/`. Third-party
code attribution is recorded in `NOTICE`.
