# OSPAN data layout

```text
ospan_sws_ter/
|-- test_image/
|-- test_annotation/             # DOTA-style polygon text files
`-- semi_ratio_20/
    `-- sparse_ratio_20/
        |-- label_image/
        |-- label_annotation/
        |-- unlabel_image/
        `-- unlabel_annotation/  # empty files for unlabeled images
```

Set `SWS_TER_DATA_ROOT` to the absolute `ospan_sws_ter` directory before
running configuration, training, or evaluation commands. The released local
split contains 452 labeled training images, 1,810 unlabeled training images,
and 565 test images. Images use the `.jpg` suffix.

The default loader reads each SAR tile as grayscale. `SWSterStudent` preserves
that single channel for PSKG/SAR evidence and repeats it only at the pretrained
ResNet input boundary. Optional Xpol/covariance preparation tools are included
but are deliberately not a prerequisite for the single-channel network path.

`a%-b%` means that `a%` of training images are initially labeled and only
`b%` of instances in those images are retained. The other training images
have empty annotation files and are used by the unlabeled branch.

The weak-supervision type is encoded in the second label column inside the
training pipeline: `0=RBox`, `1=HBox`, `2=Point`. Because the provided OSPAN
annotations are horizontal boxes, the primary OSPAN configuration sets
`hbox_proportion=1.0`. Change `point_proportion` and `hbox_proportion` only
when reproducing a different supervision experiment.

The repository excludes imagery, full annotations, checkpoints and generated
ACPC files. Reproducing numerical results requires the same train/test split;
publish the split file lists or their cryptographic hashes with the dataset.
