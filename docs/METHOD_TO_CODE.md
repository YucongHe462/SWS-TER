# Method-to-code map

This document maps the method components and equations to their implementation.
Configurable values not specified by the paper are listed in
`REPRODUCIBILITY.md`.

| Paper item | Implementation | Notes |
|---|---|---|
| Eqs. (1)-(3), `Xpol` | `tools/prepare_xpol.py` | Stacks normalized PolSARpro/MATLAB `Pd_Y4O`, `Pv_Y4O`, `Delta_Sani`. |
| Eq. (4), revised Wishart saliency | `tools/acpc/build_regions.py::compute_superpixel_responses` | Uses Pol-SLIC neighbours and complex covariance matrices supplied with `--covariance-dirs`; the implementation applies spatial weighting and a bounded transform to the Wishart distance. A grayscale covariance approximation is used when covariance files are unavailable. |
| Eqs. (5)-(7), reliable background and anchors | `tools/acpc/build_regions.py` | Enforces both `tau_sal` and `tau_var`; emits target/background/hard/uncertain seeds. |
| Eq. (8), diversity affinity | `tools/acpc/diversity_stimulation.py` | Response-morphology K-means followed by Gaussian-affinity redundancy pruning. |
| Eqs. (9)-(12), SCFE/MoCo/InfoNCE | `projects/SWS_TER/sws_ter/models/acpc.py` | 32/64/128 conv blocks, momentum encoder, queue, temperature 0.07. |
| Region prior vector `pi_reg` | `tools/acpc/sal_threshold_partition.py` | Prototype cosine similarity plus physical response; exports continuous `P_tar/P_bg/P_hard`, reliability and Eq. (32) weight. |
| Eqs. (13)-(19), SACC | `projects/SWS_TER/sws_ter/models/sacc.py` | `(k,d)={(3,1),(5,2),(7,3)}`, height-channel/width-channel/spatial interaction, compact `C/16` descriptor and competitive softmax. |
| Eqs. (20)-(21), structure response | `projects/SWS_TER/sws_ter/models/pskg.py::structure_response` | Gaussian structure tensor and Harris-like response with `zeta=0.04`. |
| Eq. (22), node feature | `projects/SWS_TER/sws_ter/models/pskg.py` | Response, normalized position and bilinearly sampled FPN feature. |
| Eqs. (23)-(25), KNN/GraphSAGE | `projects/SWS_TER/sws_ter/models/pskg.py` | Weighted spatial/response affinity and configurable GraphSAGE depth. |
| Eqs. (26)-(28), confidence/evidence/scatter | `projects/SWS_TER/sws_ter/models/pskg.py` | Component pooling, sigmoid MLP, Eq. (27) sum aggregation, Gaussian evidence and bilinear splatting. |
| Eqs. (29)-(31), EGCSF | `projects/SWS_TER/sws_ter/models/egcsf.py` | Evidence/context/structure gate, lightweight structural mapping and learnable small residual scale. |
| Eqs. (32)-(33), SALRP | `semi_mmrotate/models/dense_heads/sws_ter_head.py` and `mmdet/models/losses/sparse_focal_loss.py` | Superpixel-ID assignment; ACPC weights and the 0.4 factor are restricted to high-confidence negative locations. |
| Eq. (34), EMA | `semi_mmrotate/utils/hooks/mean_teacher.py` | Config momentum `0.9996`; begins after the 12,800-iteration burn-in. |
| Eqs. (35)-(36), confidence/GMM | `projects/SWS_TER/sws_ter/models/teacher.py::TorchGMM2` | `max(sigmoid(cls))*sigmoid(centerness)` and separate two-Gaussian EM per FPN level. |
| Eq. (37), prototype reconstruction | `projects/SWS_TER/sws_ter/models/teacher.py::PrototypeGuidedReconstructor` | Masks uncertain FPN latent tokens; high-confidence tokens and online class prototypes provide context. |
| Eqs. (38)-(40), reconstruction/distillation | `projects/SWS_TER/sws_ter/models/teacher.py` | Cosine prototype loss and detached soft-label BCE. |
| Eq. (41), unsupervised loss | `projects/SWS_TER/sws_ter/models/ugsrt_loss.py` | High-confidence classification/box/centerness plus weighted reconstruction and distillation. |
| Eqs. (42)-(46), supervised/total loss | `semi_mmrotate/models/dense_heads/sws_ter_head.py`, `mmrotate/models/losses/weak_geometry_losses.py`, teacher wrapper | Annotation-gated RBox/HBox/Point losses, centerness, superpixel/Voronoi, overlap, edge, geometric consistency and unsupervised terms. The primary config uses unit supervised/unsupervised weights. |

## Runtime data flow

1. `run_acpc.py` sees all labeled and unlabeled training images and freezes an
   SCFE checkpoint plus `acpc_priors.json`.
2. During burn-in, `SWSterStudent` builds ResNet-50/FPN features, computes the
   SACC context and PSKG structure branches, and combines them through EGCSF.
   The sparse head receives RBox/HBox/Point
   labels, while ACPC priors modulate background classification.
3. After iteration 12,800, the EMA teacher predicts weak views. UGSRT fits a
   GMM at each FPN level, distils high candidates and reconstructs uncertain
   latent features. Strong-view student predictions receive the recovered
   supervision.

Inference is explicitly dispatched to the student branch. The EMA teacher is
used only to generate training targets.

The descriptor width in Eq. (17) and ViT depth in Eq. (37) are not stated in
the prose. They are fixed to `C/16` and four layers, respectively, because
these values reproduce the parameter counts reported in Table 9.
