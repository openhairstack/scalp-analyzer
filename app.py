"""Open Hair Stack — Scalp Analyzer demo.

Upload a scalp photo -> follicular-unit detection (fu1..fu5 = hair count per
unit) -> density stats, illustrative Norwood heuristic, illustrative graft-plan
simulation.

Research/education only. Not a medical device. See MEDICAL_DISCLAIMER.md.
"""
from __future__ import annotations

import gradio as gr

import core

DISCLAIMER = (
    "**Research demo — not a medical device.** Predictions are illustrative and must not be used "
    "for diagnosis, treatment, or surgical planning. Full disclaimer: MEDICAL_DISCLAIMER.md · "
    "Project: github.com/openhairstack"
)

ABOUT = """
### What this actually does
- **Detection:** an open YOLOv11 model detects follicular units and classifies each by hair
  count (`fu1`=1, `fu2`=2–3, `fu3`=4–5, `fu4`=6+, `fu5`=unreadable).
- **Stats:** FU counts, mean hairs per unit, and — if you calibrate scale — FU/cm² density
  split across anterior/middle/posterior bands.
- **Heuristic:** a crude anterior-vs-posterior miniaturization readout. It is a heuristic, not
  a Norwood classifier.
- **Plan simulation:** a transparent, greedy allocation of grafts to deficit zones using
  openly published planning assumptions. It is a sandbox, not advice.

### Honest status
v0.1 trained on ~688 public CC-BY images. Numbers are in the model card. This is a baseline
to improve, not a product.

### Improve it
- Label images: data-pipeline repo (guide inside)
- Train better models: one-click Colab in follicle-detect repo
- File issues when it fails — failure reports are the most valuable contribution.
"""


def _status_banner() -> str:
    src, name = core.resolve_weights()
    try:
        core.get_model()
        return f"Model loaded: `{name}`"
    except core.ModelNotAvailable as exc:
        return f"**Weights not loaded** — {exc}"


def run(image, conf, px_per_mm, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a scalp photo first.")
    progress(0.2, desc="loading model")
    try:
        analysis = core.analyze(image, conf=conf, px_per_mm=px_per_mm or None)
    except core.ModelNotAvailable as exc:
        raise gr.Error(str(exc))
    progress(0.9, desc="drawing results")

    c = analysis.counts
    summary = (
        f"**{analysis.total} follicular units detected** · mean hairs/unit "
        f"**{analysis.mean_hairs_per_unit:.2f}**\n\n"
        f"`fu1` {c['fu1']} · `fu2` {c['fu2']} · `fu3` {c['fu3']} · `fu4` {c['fu4']} · `fu5`(unreadable) {c['fu5']}\n\n"
    )
    if analysis.visible_area_cm2:
        summary += (
            f"Visible area {analysis.visible_area_cm2:.1f} cm² · density "
            f"**{analysis.density_per_cm2:.1f} FU/cm²**\n\n"
        )
    for b in analysis.bands:
        dens = f", {b['density']:.1f}/cm²" if b["density"] else ""
        summary += f"- {b['name']}: {b['count']} units{dens}, {float(b['fu1_share']) * 100:.0f}% single-hair\n"
    summary += f"\n**Heuristic readout:** {analysis.norwood_heuristic}"

    rows = [
        [r["zone"], r["area_cm2"],
         "-" if r["zone_density"] is None else r["zone_density"],
         r["deficit_per_cm2"], r["grafts_target"], r["grafts_allocated"]]
        for r in analysis.plan_rows
    ]
    notes = "**Assumptions used:**\n" + "\n".join(f"- {a}" for a in analysis.assumptions)
    notes += "\n\n" + DISCLAIMER

    return analysis.annotated, summary, rows, analysis.plan_chart, notes, _status_banner()


with gr.Blocks(title="Open Hair Stack — Scalp Analyzer") as demo:
    gr.Markdown(
        "# Open Hair Stack — Scalp Analyzer\n"
        "Open-source follicular-unit detection + illustrative transplant-planning simulation. "
        "v0.1 baseline · trained on public CC-BY data · "
        "[project](https://github.com/openhairstack) · [how it works](https://github.com/openhairstack/follicle-detect)"
    )
    status = gr.Markdown(_status_banner())

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Scalp photo (trichoscope, dermoscopy, or phone macro)", type="pil")
            conf_in = gr.Slider(0.05, 0.7, value=0.25, step=0.05, label="Confidence threshold")
            scale_in = gr.Number(label="Calibration: pixels per mm (optional — measure a ruler/coin in the photo)", precision=1)
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            annotated_out = gr.Image(label="Detections", type="pil")
            summary_out = gr.Markdown()

    with gr.Row():
        with gr.Column():
            plan_out = gr.DataFrame(
                headers=["zone", "area cm²", "zone density/cm²", "deficit/cm²", "grafts target", "allocated"],
                label="Illustrative plan (sandbox — not advice)", interactive=False, row_count=3,
            )
        with gr.Column():
            chart_out = gr.Image(label="Allocation", type="pil")
    notes_out = gr.Markdown(DISCLAIMER)

    run_btn.click(run, inputs=[image_in, conf_in, scale_in],
                  outputs=[annotated_out, summary_out, plan_out, chart_out, notes_out, status])

    with gr.Accordion("About, limitations, and how to help", open=False):
        gr.Markdown(ABOUT)

    demo.load(lambda: _status_banner(), outputs=status)

demo.launch()
