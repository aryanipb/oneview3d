# OneView3D — DeepSonar point-cloud completion (web demo)

Runs entirely in your browser (onnxruntime-web, WASM): load one `_shapenet.npy` scan from the VL53L5CX ToF device (or a bundled sample) and see the observed patch plus the AdaPoinTr completion fine-tuned on device-faithful simulated scans.

Limits: a single scan sees only a small patch; the hidden side is a prediction, not a measurement; no metric size is recoverable from the exported files. Android APK: see Releases.

Model: AdaPoinTr (Yu et al., TPAMI 2023, MIT, github.com/yuxumin/PoinTr), fine-tuned.
