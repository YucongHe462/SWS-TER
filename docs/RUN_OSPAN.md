# Running every stage on OSPAN

The commands below use PowerShell and must be run from the repository root.
The dataset root is the directory that contains both `semi_ratio_20` and
`test_image`; it is not the nested `sparse_ratio_20` directory.

## 0. Create and activate the environment

```powershell
conda env create -f requirements\environment-legacy.yml
conda activate sws-ter-legacy
Set-Location 'D:\code\SWS-TER'
```

For later sessions, only `conda activate` and `Set-Location` are required.

## 1. Select and verify the data

```powershell
$DataRoot = 'D:\datasets\ospan'
$TrainSplit = Join-Path $DataRoot 'semi_ratio_20\sparse_ratio_20'
$env:SWS_TER_DATA_ROOT = $DataRoot

python tools\verify_ospan_data.py `
  --data-root $DataRoot `
  --expect-counts
```

The released split should report 452 labeled images, 1,810 unlabeled images,
565 test images, 513 sparse training objects and 1,670 test objects. The
unlabeled annotation files are intentionally empty.

## 2. Stage I: build the ACPC priors

```powershell
python tools\acpc\run_acpc.py `
  --xpol-dirs (Join-Path $TrainSplit 'label_image') `
               (Join-Path $TrainSplit 'unlabel_image') `
  --work-dir work_dirs\acpc `
  --epochs 20 `
  --batch-size 128 `
  --num-workers 0 `
  --device cuda `
  --in-channels 1 `
  --region-size 24 `
  --patch-size 96 `
  --num-clusters 128 `
  --retain-per-cluster 64

$env:SWS_TER_ACPC_PRIOR = `
  (Resolve-Path 'work_dirs\acpc\acpc_priors.json').Path
```

The wrapper executes region construction, diversity stimulation, SCFE
contrastive learning, embedding extraction and continuous prior generation.
Its final required output is `work_dirs\acpc\acpc_priors.json`.

## 3. Build the complete network before training

```powershell
python tests\framework_smoke.py
```

This constructs both branches, verifies that the frozen teacher remains in
evaluation mode, checks the single-channel forward path, SALRP gating, UGSRT,
the student inference branch and the parameter counts reported in Table 9.

## 4. Stage II: supervised burn-in

```powershell
python tools\train.py `
  configs\sws_ter\sws_ter_ospan_hbox_20_20.py `
  --work-dir work_dirs\sws_ter_burnin `
  --no-validate `
  --cfg-options `
    runner.max_iters=12800 `
    checkpoint_config.interval=12800
```

Iterations 1-12,800 use only the labeled branch. The expected checkpoint is
`work_dirs\sws_ter_burnin\iter_12800.pth`.

## 5. Stage III: teacher-student recovery training

```powershell
python tools\train.py `
  configs\sws_ter\sws_ter_ospan_hbox_20_20.py `
  --work-dir work_dirs\sws_ter_full `
  --resume-from work_dirs\sws_ter_burnin\iter_12800.pth
```

The first resumed iteration initializes the EMA teacher from the trained
student. Training then continues through iteration 48,000 with
`L_total = L_sup + L_unsup`. Standard `iter_N.pth` checkpoints continue at
iteration `N` without repeating the stored optimization step.

Stages II and III can also run without an intermediate stop:

```powershell
python tools\train.py `
  configs\sws_ter\sws_ter_ospan_hbox_20_20.py `
  --work-dir work_dirs\sws_ter_full
```

## 6. Student-branch inference and evaluation

```powershell
python tools\test.py `
  configs\sws_ter\sws_ter_ospan_hbox_20_20.py `
  work_dirs\sws_ter_full\latest.pth `
  --eval mAP `
  --work-dir work_dirs\sws_ter_full\evaluation
```

The configuration sets `inference_on='student'`; the EMA teacher is not used
to produce the reported detections.

## Optional short end-to-end check

The following two training commands exercise one burn-in iteration, resume,
one teacher-student iteration and checkpoint saving at 128 x 128. They are a
runtime check only and do not reproduce reported accuracy.

```powershell
$SmokeOptions = @(
  'model.model.backbone.init_cfg=None',
  'data.samples_per_gpu=4',
  'data.workers_per_gpu=0',
  'data.train.sup.pipeline.3.img_scale=(128,128)',
  'data.train.sup.pipeline.3.ratio_range=(1.0,1.0)',
  'data.train.unsup_unlabeled.pipeline.2.unsup_weak.0.img_scale=(128,128)',
  'data.train.unsup_unlabeled.pipeline.2.unsup_weak.0.ratio_range=(1.0,1.0)',
  'model.train_cfg.burn_in_steps=1',
  'custom_hooks.1.start_steps=1',
  'checkpoint_config.interval=1',
  'log_config.interval=1'
)

python tools\train.py `
  configs\sws_ter\sws_ter_ospan_hbox_20_20.py `
  --work-dir work_dirs\runtime_check_burnin `
  --no-validate `
  --cfg-options @SmokeOptions runner.max_iters=1

python tools\train.py `
  configs\sws_ter\sws_ter_ospan_hbox_20_20.py `
  --work-dir work_dirs\runtime_check_full `
  --resume-from work_dirs\runtime_check_burnin\iter_1.pth `
  --no-validate `
  --cfg-options @SmokeOptions runner.max_iters=2
```
