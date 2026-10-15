# scalp-analyzer

Free, browser-based follicular-unit analyzer for scalp photos — the public face of the Open Hair Stack.

Upload a photo → get detected follicular units (colored by hair count: `fu1`=1 … `fu5`=unreadable) → density stats → an **illustrative** graft-planning sandbox.

Live demo: https://huggingface.co/spaces/openhairstack/scalp-analyzer

## What's real and what's simulated

| Feature | Status |
|---|---|
| Follicle detection + hair-count class | real — YOLOv11 baseline `fu-v0.1`, metrics in model card |
| FU counts, mean hairs/unit | real arithmetic on detections |
| FU/cm² density | real, only when you calibrate scale (pixels per mm) |
| Norwood-style heuristic | **heuristic** — anterior/posterior miniaturization ratio, not a trained classifier |
| Graft allocation | **simulation** — greedy proportional allocation on stated assumptions (target 25 FU/cm², 30% donor harvest, assumed zone areas) |

Every number the app prints beyond detection is a function you can read in `core.py`. There is no black box in the demo — on purpose.

## Run it

### Local
```bash
pip install -r requirements.txt
# needs weights: train via the Colab notebook in openhairstack/follicle-detect,
# then either set LOCAL_WEIGHTS=path/to/best.pt or MODEL_ID=<hf repo id>
python app.py
```
Without weights the app loads and shows a clear status message instead of erroring.

### Weights resolution order
1. `LOCAL_WEIGHTS` env var (a path to `best.pt`)
2. `MODEL_ID` env var (a public Hugging Face repo; defaults to `openhairstack/follicle-fu-v0.1`)
3. bundled `weights/best.pt` if present

### Hugging Face Space
This repo is Space-ready (Gradio SDK, CPU-basic tier is enough: ~5–15 s/image).
Set the `MODEL_ID` secret in Space settings to your trained model's repo.

## Architecture (intentionally boring)

```
app.py     UI wiring, event handlers
core.py    detection, stats, bands, heuristic, plan simulation, PIL chart (no matplotlib)
```
No server side. No accounts. No telemetry. Uploads are processed in memory and never stored.

## Roadmap for the demo

- v0.1.1 calibrated ruler overlay in the UI for scale entry (two-click)
- v0.2 OBB boxes (aligned to follicle axis — better crowding)
- v0.3 consent/synthetic sample gallery
- v1.0 "explain" mode: highlight what the model sees vs. what a clinician would

## Health warning

This is research software. It is not a medical device, it is not FDA/CE cleared, and it must
not inform anyone's treatment. See [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md).
