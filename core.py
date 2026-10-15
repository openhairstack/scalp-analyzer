"""Core analysis pipeline for the scalp-analyzer demo.

Pipeline: detect follicular units -> stats (counts, density bands) ->
Norwood-heuristic readout -> illustrative graft-planning simulation.

Everything here runs on CPU. No network needed at inference time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

CANON = ["fu1", "fu2", "fu3", "fu4", "fu5"]
DEFAULT_MODEL_ID = "openhairstack/follicle-fu-v0.1"
AVG_HAIRS_PER_UNIT = {"fu1": 1.0, "fu2": 2.0, "fu3": 3.0, "fu4": 4.0, "fu5": 5.0}  # v0.1 classes = exact hair count
CLASS_COLORS = {
    "fu1": (230, 25, 75), "fu2": (245, 130, 48), "fu3": (60, 180, 75),
    "fu4": (0, 130, 200), "fu5": (128, 128, 128),
}


class ModelNotAvailable(RuntimeError):
    pass


def resolve_weights():
    """Local weights file or HF repo id. Returns (model_id_or_path, display_name)."""
    local = os.environ.get("LOCAL_WEIGHTS", "weights/best.pt")
    if os.path.exists(local):
        return local, os.path.basename(local)
    return os.environ.get("MODEL_ID", DEFAULT_MODEL_ID), os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)


def get_model():
    from ultralytics import YOLO

    src, name = resolve_weights()
    if os.path.isfile(src):
        return YOLO(src), name
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id=src)
    except Exception as exc:  # noqa: BLE001
        raise ModelNotAvailable(
            f"weights repo '{src}' not reachable ({exc.__class__.__name__}). "
            "Train with the Colab notebook in openhairstack/follicle-detect and publish, "
            "then set the MODEL_ID setting to your repo."
        ) from exc
    pts = sorted(Path(path).rglob("*.pt"))
    if not pts:
        raise ModelNotAvailable(f"no .pt weights inside {src}")
    return YOLO(str(pts[-1])), name


@dataclass
class Analysis:
    annotated: object
    counts: dict
    total: int
    mean_hairs_per_unit: float
    scale_known: bool
    visible_area_cm2: float | None
    density_per_cm2: float | None
    bands: list          # [{name, fu_count, fu1_share, area_cm2|None, density|None}]
    norwood_heuristic: str
    plan_rows: list      # rows for gr.DataFrame
    plan_chart: object   # PIL image
    assumptions: list = field(default_factory=list)


def _bar_chart(plan_rows):
    from PIL import Image, ImageDraw

    W, H, ROW = 640, 60 + 46 * len(plan_rows), 46
    img = Image.new("RGB", (W, H), (16, 20, 28))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 46], fill=(30, 36, 50))
    d.text((16, 14), "Illustrative graft allocation (NOT a surgical plan)", fill=(230, 230, 235))
    max_v = max((r["grafts_target"] for r in plan_rows), default=1) or 1
    y = 46
    for r in plan_rows:
        bg_w = 300
        t_w = int(bg_w * r["grafts_target"] / max_v)
        a_w = int(bg_w * r["grafts_allocated"] / max_v)
        d.text((16, y + 4), r["zone"], fill=(200, 205, 215))
        d.rectangle([170, y + 6, 170 + bg_w, y + 18], outline=(70, 78, 95))
        d.rectangle([170, y + 6, 170 + a_w, y + 18], fill=(60, 180, 75))
        d.rectangle([170, y + 20, 170 + t_w, y + 32], fill=(0, 130, 200))
        d.text((486, y + 8), f'{r["grafts_allocated"]}/{r["grafts_target"]}', fill=(220, 220, 225))
        y += ROW
    d.text((170, H - 26), "green = allocated   blue = zone target", fill=(140, 148, 165))
    return img


def plan_simulation(bands, density_per_cm2, scale_known, total_fu):
    """Greedy proportional allocation. Transparent, crude, illustrative by design."""
    ZONE_AREAS = {"anterior": 45.0, "middle": 45.0, "posterior": 40.0}  # cm2, assumed
    TARGET = 25.0          # target FU/cm2 density in treated zones
    DONOR_AREA = 150.0     # cm2 of usable donor zone (typical)
    HARVEST_FRACTION = 0.30 # fraction of donor FUs harvested per session (conservative)

    if density_per_cm2:
        donor_capacity = int(density_per_cm2 * DONOR_AREA * HARVEST_FRACTION)
        supply_note = f"donor capacity est. {donor_capacity:.0f} FU (measured density {density_per_cm2:.1f}/cm2)"
        zone_areas = {b["name"]: (b["area_cm2"] or ZONE_AREAS[b["name"]]) for b in bands}
    else:
        donor_capacity = int(25 * DONOR_AREA * HARVEST_FRACTION)  # assume typical 25/cm2 donor
        supply_note = "donor capacity assumed ~25 FU/cm2 x 150 cm2 x 30% harvest"
        zone_areas = ZONE_AREAS

    rows = []
    demand_total = 0.0
    for b in bands:
        area = zone_areas[b["name"]]
        zone_density = b["density"] or (b["count"] / max(area, 1e-6) if scale_known else None)
        deficit = max(0.0, TARGET - (zone_density if zone_density is not None else TARGET * 0.8))
        need = int(area * deficit)
        demand_total += need
        rows.append({"zone": b["name"].title(), "area_cm2": round(area, 1),
                     "zone_density": None if zone_density is None else round(zone_density, 1),
                     "deficit_per_cm2": round(deficit, 1), "grafts_target": need,
                     "grafts_allocated": 0})

    scale = min(1.0, donor_capacity / demand_total) if demand_total > 0 else 1.0
    for r in rows:
        r["grafts_allocated"] = int(r["grafts_target"] * scale)
    return rows, donor_capacity, supply_note, demand_total


def analyze(image, conf: float = 0.25, px_per_mm: float | None = None) -> Analysis:
    from PIL import Image, ImageDraw

    model, model_name = get_model()
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    results = model.predict(image, conf=conf, imgsz=640, verbose=False)
    r = results[0]
    names = r.names

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        cls = names.get(cls_id, str(cls_id))
        xyxy = [int(v) for v in box.xyxy.cpu().numpy().tolist()[0]]
        detections.append({"class": cls, "conf": float(box.conf.item()), "xyxy": xyxy,
                           "center": ((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)})

    counts = {c: 0 for c in CANON}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    total = len(detections)
    mean_hairs = (sum(counts[c] * AVG_HAIRS_PER_UNIT[c] for c in AVG_HAIRS_PER_UNIT) / total) if total else 0.0

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for d in detections:
        c = CLASS_COLORS.get(d["class"], (200, 200, 200))
        draw.rectangle(d["xyxy"], outline=c, width=3)
        draw.text((d["xyxy"][0], max(0, d["xyxy"][1] - 12)), f'{d["class"]} {float(d["conf"]):.2f}', fill=c)
    ly = 6
    for cls, n in counts.items():
        draw.rectangle([6, ly, 14, ly + 8], fill=CLASS_COLORS.get(cls, (128, 128, 128)))
        draw.text((18, ly - 2), f"{cls}: {n}", fill=(255, 255, 255))
        ly += 14

    w, h = annotated.size
    scale_known = px_per_mm is not None and px_per_mm > 3
    visible_area_cm2 = (w * h / (px_per_mm ** 2) / 100.0) if scale_known else None
    density = (total / visible_area_cm2) if visible_area_cm2 else None

    band_defs = [("anterior", 0), ("middle", 1), ("posterior", 2)]
    bands = []
    for band, idx in band_defs:
        y0, y1 = int(h * idx / 3), int(h * (idx + 1) / 3)
        band_dets = [d for d in detections if y0 <= d["center"][1] < y1]
        band_area = (w * (y1 - y0) / (px_per_mm ** 2) / 100.0) if scale_known else None
        band_density = (len(band_dets) / band_area) if band_area else None
        fu1_share = (sum(1 for d in band_dets if d["class"] == "fu1") / len(band_dets)) if band_dets else 0.0
        bands.append({"name": band, "count": len(band_dets), "fu1_share": fu1_share,
                      "area_cm2": band_area, "density": band_density})

    post = bands[2]
    ant = bands[0]
    if post["count"] == 0 or ant["count"] == 0:
        norwood = "inconclusive (crop to include donor area at image bottom)"
    else:
        ratio = (ant["fu1_share"] + 0.05) / (post["fu1_share"] + 0.05)
        if ratio > 1.6 or ant["count"] < post["count"] * 0.6:
            norwood = "pattern consistent with significant frontal thinning (illustrative: NW ~3-4)"
        elif ratio > 1.25 or ant["count"] < post["count"] * 0.85:
            norwood = "possible frontal thinning (illustrative: NW ~2)"
        else:
            norwood = "no obvious anterior/posterior gradient in this crop (illustrative: NW 1)"

    plan_rows, capacity, supply_note, demand = plan_simulation(bands, density, scale_known, total)

    assumptions = []
    if not scale_known:
        assumptions.append("no scale calibration: zone areas assumed 45/45/40 cm2, targets illustrative")
    assumptions.append(f"{supply_note}")
    assumptions.append("target density 25 FU/cm2 in treated zones; 30% donor harvest fraction")

    return Analysis(
        annotated=annotated, counts=counts, total=total, mean_hairs_per_unit=mean_hairs,
        scale_known=scale_known, visible_area_cm2=visible_area_cm2, density_per_cm2=density,
        bands=bands, norwood_heuristic=norwood, plan_rows=plan_rows,
        plan_chart=_bar_chart(plan_rows), assumptions=assumptions,
    )
