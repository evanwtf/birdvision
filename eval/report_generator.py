#!/usr/bin/env python3
"""
BirdVision model comparison report generator.

Reads all eval sidecar JSONs written by eval_runner.py and produces a
self-contained HTML report: report/report.html.

Crop images are copied into report/crops/ and referenced with relative
paths so the report directory is portable.

Usage:
    uv run eval/report_generator.py --config eval/config.yaml
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("report_generator")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sidecars(eval_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {asset_sha: {model_id: sidecar_dict}}."""
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
            continue
        sha = data.get("asset_sha")
        mid = data.get("model_id")
        if sha and mid:
            result[sha][mid] = data
    return dict(result)


def load_result_meta(results_dir: Path, sidecar: dict) -> dict:
    """Load metadata (date, source filename, stills) from the original results JSON."""
    src = sidecar.get("source_results_file", "")
    result_file = results_dir / src
    if not result_file.exists():
        return {}
    try:
        d = json.loads(result_file.read_text())
        return {
            "date": d.get("date", ""),
            "source_filename": d.get("source_filename") or d.get("display_name", ""),
            "stills": [
                s["file"] for s in d.get("frame_gallery", [])
                if "annotated" in s.get("file", "")
            ][:1],
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------

def _top1(sidecar: dict) -> str:
    for track in sidecar.get("tracks", []):
        preds = track.get("top_species", [])
        if preds:
            return preds[0]["species"]
    return ""


def _top_preds_html(sidecar: dict) -> str:
    rows = []
    for track in sidecar.get("tracks", []):
        preds = track.get("top_species", [])[:5]
        if not preds:
            continue
        items = " &nbsp;|&nbsp; ".join(
            f'<span class="sp">{p["species"]}</span> <span class="sc">{p["score"]:.1%}</span>'
            for p in preds
        )
        rows.append(f'<div class="track-preds">Track {track["track_id"]}: {items}</div>')
    return "\n".join(rows) if rows else '<span class="muted">no predictions</span>'


IMPLAUSIBLE_CSS_CLASS = "implausible"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(cfg: dict) -> None:
    results_dir = Path(cfg["results_dir"])
    report_dir = Path(cfg["report_dir"])
    eval_dir = report_dir / "eval"
    crops_out_dir = report_dir / "crops"
    crops_out_dir.mkdir(parents=True, exist_ok=True)

    implausible = set(cfg.get("implausible_at_feeder", []))
    model_cfgs = cfg.get("models", [])
    model_ids = [m["id"] for m in model_cfgs]
    model_labels = {m["id"]: m.get("label", m["id"]) for m in model_cfgs}

    all_data = load_sidecars(eval_dir)
    if not all_data:
        logger.error("No sidecar files found in %s — run eval_runner.py first", eval_dir)
        return

    logger.info("Loaded sidecars for %d assets", len(all_data))

    # Build per-asset rows
    rows = []
    for asset_sha, models in all_data.items():
        # Only include assets that have at least one model's data
        if not models:
            continue

        # Get metadata from any available sidecar
        first_sidecar = next(iter(models.values()))
        meta = load_result_meta(results_dir, first_sidecar)

        crops_dir = results_dir / f"{asset_sha}_crops"

        # Collect crop images (first track crop from each asset)
        crop_img_rel = None
        for track in first_sidecar.get("tracks", []):
            crop_file = track.get("crop_file", "")
            src_crop = crops_dir / crop_file
            if crop_file and src_crop.exists():
                dest_name = f"{asset_sha[:16]}_{crop_file}"
                dest = crops_out_dir / dest_name
                if not dest.exists():
                    shutil.copy2(src_crop, dest)
                crop_img_rel = f"crops/{dest_name}"
                break

        # Also copy an annotated still if available
        still_img_rel = None
        for still_file in meta.get("stills", []):
            src_still = crops_dir / still_file
            if src_still.exists():
                dest_name = f"{asset_sha[:16]}_{still_file}"
                dest = crops_out_dir / dest_name
                if not dest.exists():
                    shutil.copy2(src_still, dest)
                still_img_rel = f"crops/{dest_name}"
                break

        # Per-model top-1 species
        top1_by_model = {mid: _top1(sc) for mid, sc in models.items()}

        # Agreement: all enabled models with data agree on top-1
        present_top1 = [s for s in top1_by_model.values() if s]
        agrees = len(set(present_top1)) <= 1 if present_top1 else True

        # Any implausible top-1?
        has_implausible = any(s in implausible for s in top1_by_model.values() if s)

        rows.append({
            "asset_sha": asset_sha,
            "date": meta.get("date", ""),
            "source_filename": meta.get("source_filename", ""),
            "models": models,
            "top1_by_model": top1_by_model,
            "agrees": agrees,
            "has_implausible": has_implausible,
            "crop_img": crop_img_rel,
            "still_img": still_img_rel,
        })

    # Sort: disagreements first, then implausible, then rest
    rows.sort(key=lambda r: (r["agrees"], not r["has_implausible"], r["date"]))

    # Summary counts
    n_total = len(rows)
    n_disagree = sum(1 for r in rows if not r["agrees"])
    n_implausible = sum(1 for r in rows if r["has_implausible"])

    # ---- Render HTML -------------------------------------------------------
    html = _render_html(
        rows=rows,
        model_ids=model_ids,
        model_labels=model_labels,
        implausible=implausible,
        n_total=n_total,
        n_disagree=n_disagree,
        n_implausible=n_implausible,
    )

    out = report_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Report written: %s  (%d clips)", out, len(rows))


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _render_html(
    rows: list[dict],
    model_ids: list[str],
    model_labels: dict[str, str],
    implausible: set[str],
    n_total: int,
    n_disagree: int,
    n_implausible: int,
) -> str:
    cards_html = "\n".join(_render_card(r, model_ids, model_labels, implausible) for r in rows)
    table_html = _render_summary_table(rows, model_ids, model_labels, implausible)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BirdVision Model Comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1rem 2rem; background: #f5f5f5; color: #222; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .stats {{ color: #555; margin-bottom: 1.5rem; font-size: 0.95rem; }}
  .stats span {{ margin-right: 1.5rem; }}
  .badge-disagree {{ background: #fce8b2; border-radius: 3px; padding: 1px 5px; }}
  .badge-implausible {{ background: #f4c2c2; border-radius: 3px; padding: 1px 5px; }}

  /* Filters */
  .filters {{ margin-bottom: 1.5rem; }}
  .filters label {{ margin-right: 1rem; font-size: 0.9rem; cursor: pointer; }}

  /* Summary table */
  details {{ margin-bottom: 2rem; }}
  summary {{ cursor: pointer; font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; background: #fff; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; }}
  th {{ background: #eee; }}
  tr.implausible td:nth-child(3) {{ background: #ffd6d6; }}
  tr.disagree {{ background: #fffde7; }}

  /* Cards */
  .cards {{ display: flex; flex-direction: column; gap: 1rem; }}
  .card {{ background: #fff; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.15); padding: 1rem; display: grid; grid-template-columns: 160px 160px 1fr; gap: 1rem; align-items: start; }}
  .card.disagree {{ border-left: 4px solid #f0a500; }}
  .card.implausible {{ border-left: 4px solid #d32f2f; }}
  .card img {{ width: 150px; height: 110px; object-fit: cover; border-radius: 4px; background: #ccc; display: block; }}
  .card .no-img {{ width: 150px; height: 110px; background: #ddd; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #888; font-size: 0.75rem; }}
  .card-meta {{ font-size: 0.75rem; color: #666; margin-top: 0.3rem; }}
  .card-models {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .model-block {{ font-size: 0.85rem; }}
  .model-name {{ font-weight: 600; color: #444; margin-bottom: 0.15rem; }}
  .track-preds {{ margin-bottom: 0.1rem; }}
  .sp {{ color: #1a237e; }}
  .sc {{ color: #666; }}
  .top1-implausible .sp:first-child {{ color: #c62828; font-weight: bold; }}
  .muted {{ color: #aaa; font-size: 0.8rem; }}
  .tag {{ display: inline-block; font-size: 0.7rem; padding: 1px 5px; border-radius: 3px; margin-left: 4px; }}
  .tag-disagree {{ background: #fce8b2; color: #6d4c00; }}
  .tag-implausible {{ background: #ffcdd2; color: #b71c1c; }}

  @media (max-width: 700px) {{
    .card {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<h1>BirdVision Model Comparison</h1>
<div class="stats">
  <span>{n_total} clips</span>
  <span class="badge-disagree">{n_disagree} disagreements</span>
  <span class="badge-implausible">{n_implausible} implausible top picks</span>
</div>

<div class="filters">
  <strong>Show:</strong>
  <label><input type="radio" name="filter" value="all" checked> All</label>
  <label><input type="radio" name="filter" value="disagree"> Disagreements only</label>
  <label><input type="radio" name="filter" value="implausible"> Implausible only</label>
</div>

<details>
  <summary>Summary table ({n_total} clips)</summary>
  {table_html}
</details>

<div class="cards" id="cards">
{cards_html}
</div>

<script>
document.querySelectorAll('input[name=filter]').forEach(r => r.addEventListener('change', () => {{
  const v = document.querySelector('input[name=filter]:checked').value;
  document.querySelectorAll('.card').forEach(c => {{
    const show = v === 'all'
      || (v === 'disagree' && c.classList.contains('disagree'))
      || (v === 'implausible' && c.classList.contains('implausible'));
    c.style.display = show ? '' : 'none';
  }});
}});
</script>
</body>
</html>"""


def _render_summary_table(
    rows: list[dict],
    model_ids: list[str],
    model_labels: dict[str, str],
    implausible: set[str],
) -> str:
    header_cols = "".join(f"<th>{model_labels.get(m, m)}</th>" for m in model_ids)
    header = f"<tr><th>Asset</th><th>Date</th>{header_cols}<th>Agreement</th></tr>"

    body_rows = []
    for r in rows:
        css = ""
        if r["has_implausible"]:
            css = "implausible"
        elif not r["agrees"]:
            css = "disagree"

        cols = "".join(
            f'<td class="{"top1-implausible" if r["top1_by_model"].get(m) in implausible else ""}">'
            f'{r["top1_by_model"].get(m, "—")}</td>'
            for m in model_ids
        )
        agree_cell = "&#10003;" if r["agrees"] else '<span style="color:#c00">&#10007;</span>'
        body_rows.append(
            f'<tr class="{css}"><td><code>{r["asset_sha"][:12]}</code></td>'
            f'<td>{r["date"][:10]}</td>{cols}<td>{agree_cell}</td></tr>'
        )

    return f"<table><thead>{header}</thead><tbody>{''.join(body_rows)}</tbody></table>"


def _render_card(
    row: dict,
    model_ids: list[str],
    model_labels: dict[str, str],
    implausible: set[str],
) -> str:
    css_classes = ["card"]
    tags = ""
    if not row["agrees"]:
        css_classes.append("disagree")
        tags += '<span class="tag tag-disagree">disagreement</span>'
    if row["has_implausible"]:
        css_classes.append("implausible")
        tags += '<span class="tag tag-implausible">implausible pick</span>'

    crop_html = (
        f'<img src="{row["crop_img"]}" alt="crop" loading="lazy">'
        if row["crop_img"] else
        '<div class="no-img">no crop</div>'
    )
    still_html = (
        f'<img src="{row["still_img"]}" alt="still" loading="lazy">'
        if row["still_img"] else
        '<div class="no-img">no still</div>'
    )

    models_html = ""
    for mid in model_ids:
        sidecar = row["models"].get(mid)
        label = model_labels.get(mid, mid)
        if sidecar:
            top1_sp = row["top1_by_model"].get(mid, "")
            implausible_cls = "top1-implausible" if top1_sp in implausible else ""
            preds = _top_preds_html(sidecar)
            models_html += (
                f'<div class="model-block">'
                f'<div class="model-name">{label}</div>'
                f'<div class="{implausible_cls}">{preds}</div>'
                f'</div>'
            )
        else:
            models_html += (
                f'<div class="model-block">'
                f'<div class="model-name">{label}</div>'
                f'<div class="muted">no data</div>'
                f'</div>'
            )

    meta = f'{row["date"][:10]} &nbsp; <code title="{row["asset_sha"]}">{row["asset_sha"][:12]}...</code>{tags}'

    return (
        f'<div class="{" ".join(css_classes)}" data-agrees="{row["agrees"]}" '
        f'data-implausible="{row["has_implausible"]}">'
        f'{crop_html}'
        f'{still_html}'
        f'<div>'
        f'<div class="card-meta">{meta}</div>'
        f'<div class="card-models">{models_html}</div>'
        f'</div>'
        f'</div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdVision model comparison report generator")
    parser.add_argument("--config", default="eval/config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    build_report(cfg)


if __name__ == "__main__":
    main()
