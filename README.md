---
title: Open Hair Stack — Scalp Analyzer
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Open-source follicular-unit detection and density analysis. Research demo.
---

# scalp-analyzer

> **The public face of Open Hair Stack:** a free website tool that analyzes scalp photos — no
> account, nothing stored, and every number it prints traces to code you can read.

### In plain English

Upload a photo of your scalp. The tool finds the little bundles of hair ("follicular units"),
counts how many hairs are in each bundle, and estimates how dense the area is. It also shows
an explicitly labeled *illustration* of how a transplant plan might distribute grafts —
that part is a sandbox for learning, not advice.

**This is not a medical device and never will be.** It doesn't diagnose anything. Full
honesty, in writing: [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md)

> **Live demo note:** the public demo is the static, browser-only version — free to host forever,
> no uploads, works on phones: **https://huggingface.co/spaces/openhairstack/scalp-analyzer**
> (source: [scalp-analyzer-web](https://github.com/openhairstack/scalp-analyzer-web)).
> This repo is the original Gradio/Python version — still fully functional for local use and
> self-hosting on any CPU machine.

---

## What's real and what's simulated

| Feature | Status |
|---|---|
| Follicle detection + hair-count class | real — YOLOv11 baseline `fu-v0.1`, metrics in [model card](https://github.com/openhairstack/follicle-detect) |
| FU counts, mean hairs/unit | real arithmetic on detections |
| FU/cm² density | real, only when you calibrate scale (pixels per mm) |
| Norwood-style heuristic | **heuristic** — anterior/posterior miniaturization ratio, not a trained classifier |
| Graft allocation | **simulation** — greedy proportional allocation on stated assumptions (target 25 FU/cm², 30% donor harvest, assumed zone areas) |

Every number beyond detection is a function you can read in `core.py`. There is no black box
in the demo — on purpose.

## Try it locally

```bash
pip install -r requirements.txt
# weights: train via the Colab notebook in openhairstack/follicle-detect,
# then either set LOCAL_WEIGHTS=path/to/best.pt or MODEL_ID=<hf repo id>
python app.py
```

Without weights, the app loads and shows a clear status message instead of erroring.

### Weights resolution order

1. `LOCAL_WEIGHTS` env var (a path to `best.pt`)
2. `MODEL_ID` env var (a public Hugging Face repo; default `openhairstack/follicle-fu-v0.1`)
3. bundled `weights/best.pt` if present

### Hugging Face Space

This repo is Space-ready (Gradio SDK; CPU-basic tier is enough, ~5–15 s/image). Set the
`MODEL_ID` secret in Space settings to your trained model repo.

## Architecture (intentionally boring)

```
app.py     UI wiring, event handlers
core.py    detection, stats, bands, heuristic, plan simulation, PIL chart (no matplotlib)
```

No server side. No accounts. No telemetry. Uploads are processed in memory and never stored.

## Technical notes and limits

- **Resolution matters.** Macro/trichoscope photos work best; wide-angle phone photos rarely
  show follicle structure. The model will fail on them, honestly.
- **Orient your crop** same as the training data (anterior at top of the frame).
- **fu5 is deliberate.** The "can't tell" class exists so the model can express uncertainty —
  don't treat it as noise.
- **Calibration is optional but honest.** Without a scale, density numbers are relative-only.
- v0.1 was trained on ~688 public images. Hair types, curls, and skin tones are
  underrepresented — that's the community's first mission, not a footnote.

## Roadmap

- **v0.1.1** two-click ruler calibration in the UI
- **v0.2** oriented boxes (OBB) for crowded follicles
- **v0.3** consented/synthetic sample gallery
- **v1.0** "explain" mode: what the model sees vs. what a clinician sees

## Health warning

Research software. Not a medical device, not FDA/CE cleared, must not inform treatment.
See [MEDICAL_DISCLAIMER.md](MEDICAL_DISCLAIMER.md).
