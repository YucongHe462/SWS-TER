# SWS-TER project extension

`sws_ter/models` contains framework-independent implementations of ACPC/SCFE,
SACC, PSKG, EGCSF and UGSRT.  `detectors.py` and `ugsrt_loss.py` are the thin
MMRotate adapters.  Keeping the mathematical core independent makes it
possible to test the method without compiling MMCV CUDA operators.

