from __future__ import annotations

import base64
import html
import io
import re

import pandas as pd

from matplotlib.figure import Figure
from matplotlib.patches import Patch


def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "run"


def df_to_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    use = df if max_rows is None else df.head(max_rows)
    return use.where(pd.notna(use), None).to_dict(orient="records")


def results_to_saved_payload(results: dict) -> dict:
    bins_df = results.get("bins", pd.DataFrame())
    shap_imp = results.get("shap_importance", {}) or {}
    shap_json = {}
    if isinstance(shap_imp, dict):
        for k, v in shap_imp.items():
            if isinstance(v, pd.DataFrame):
                shap_json[str(k)] = df_to_records(v, max_rows=500)

    roc_json = []
    for r in results.get("roc", []) or []:
        roc_json.append(
            {
                "class": str(r.get("class", "")),
                "fpr": [float(x) for x in list(r.get("fpr", []))],
                "tpr": [float(x) for x in list(r.get("tpr", []))],
                "auc": float(r.get("auc", 0.0)),
            }
        )

    metrics = {
        str(k): float(v) if isinstance(v, (int, float)) else v
        for k, v in (results.get("metrics", {}) or {}).items()
    }

    return {
        "model_name": str(results.get("model_name", "")),
        "metrics": metrics,
        "classification_report": results.get("classification_report", {}) or {},
        "classification_report_text": results.get("classification_report_text", ""),
        "meta": results.get("meta", {}) or {},
        "bins": df_to_records(bins_df, max_rows=None),
        "roc": roc_json,
        "shap_importance": shap_json,
        "visual_plots": results.get("visual_plots", {}) or {},
        "dataframe": df_to_records(
            results.get("dataframe", pd.DataFrame()), max_rows=1000
        ),
    }


def saved_payload_to_results(payload: dict) -> dict:
    shap_out = {}
    for k, rows in (payload.get("shap_importance", {}) or {}).items():
        shap_out[str(k)] = pd.DataFrame(rows or [])

    return {
        "model_name": payload.get("model_name", "Saved Run"),
        "metrics": payload.get("metrics", {}) or {},
        "classification_report": payload.get("classification_report", {}) or {},
        "classification_report_text": payload.get("classification_report_text", ""),
        "meta": payload.get("meta", {}) or {},
        "bins": pd.DataFrame(payload.get("bins", []) or []),
        "roc": payload.get("roc", []) or [],
        "shap_importance": shap_out,
        "visual_plots": payload.get("visual_plots", {}) or {},
        "dataframe": pd.DataFrame(payload.get("dataframe", []) or []),
    }


def fig_to_base64_png(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_saved_run_charts(results: dict) -> dict[str, str]:
    charts: dict[str, str] = {}

    bins = results.get("bins", pd.DataFrame())
    if (
        isinstance(bins, pd.DataFrame)
        and not bins.empty
        and {"bin_name", "count"}.issubset(bins.columns)
    ):
        fig = Figure(figsize=(7.2, 3.4), dpi=110)
        ax = fig.add_subplot(111)
        ax.bar(
            bins["bin_name"].astype(str),
            bins["count"].astype(float),
            color="#13a8f1",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        fig.tight_layout()
        charts["distribution"] = fig_to_base64_png(fig)

    roc_list = results.get("roc", []) or []
    if roc_list:
        fig = Figure(figsize=(7.2, 3.4), dpi=110)
        ax = fig.add_subplot(111)
        for r in roc_list[:6]:
            try:
                ax.plot(
                    r.get("fpr", []),
                    r.get("tpr", []),
                    label=f"{r.get('class', '')} (AUC={float(r.get('auc', 0.0)):.3f})",
                )
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], "--", linewidth=1, color="#777")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        if ax.lines:
            ax.legend(fontsize=8)
        fig.tight_layout()
        charts["roc"] = fig_to_base64_png(fig)

    visual = results.get("visual_plots", {}) or {}
    hist_items = visual.get("top_shap_hist", []) or []
    if hist_items:
        n = min(5, len(hist_items))
        cols = 3 if n > 3 else n
        rows = 2 if n > 3 else 1
        fig = Figure(figsize=(8.6, 4.6), dpi=110)
        for i, item in enumerate(hist_items[:5]):
            ax = fig.add_subplot(rows, cols, i + 1)
            pass_vals = item.get("pass_values", []) or []
            fail_vals = item.get("fail_values", []) or []
            if pass_vals:
                ax.hist(
                    pass_vals,
                    bins=24,
                    density=True,
                    alpha=0.55,
                    color="#26bdfd",
                    label="PASS",
                )
            if fail_vals:
                ax.hist(
                    fail_vals,
                    bins=24,
                    density=True,
                    alpha=0.55,
                    color="#ff7d75",
                    label="FAIL",
                )
            title = str(item.get("feature", "feature"))
            if len(title) > 28:
                title = title[:25] + "..."
            ax.set_title(title, fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)
        fig.tight_layout()
        charts["top_hist"] = fig_to_base64_png(fig)

    records = visual.get("probability_box", []) or []
    if records:
        bdf = pd.DataFrame(records)
        if "group" not in bdf.columns and "cal_group" in bdf.columns:
            bdf["group"] = bdf["cal_group"]
        if {"group", "pass_fail", "probability"}.issubset(bdf.columns):
            fig = Figure(figsize=(7.2, 3.4), dpi=110)
            ax = fig.add_subplot(111)
            group_order = bdf["group"].dropna().astype(str).drop_duplicates().tolist()
            pos, vals, colors = [], [], []
            offset = {"PASS": -0.16, "FAIL": 0.16}
            color_map = {"PASS": "#2fc1ff", "FAIL": "#ff7b72"}
            for i, grp in enumerate(group_order):
                base = i + 1
                for pf in ["PASS", "FAIL"]:
                    arr = (
                        bdf[
                            (bdf["group"].astype(str) == grp)
                            & (bdf["pass_fail"].astype(str) == pf)
                        ]["probability"]
                        .dropna()
                        .to_numpy()
                    )
                    if len(arr) == 0:
                        continue
                    vals.append(arr)
                    pos.append(base + offset[pf])
                    colors.append(color_map[pf])
            if vals:
                bp = ax.boxplot(
                    vals, positions=pos, widths=0.28, patch_artist=True, showfliers=False
                )
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.6)
                ax.set_xticks([i + 1 for i in range(len(group_order))])
                ax.set_xticklabels(group_order)
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel("Distribution Group")
                ax.set_ylabel("Probability")
                ax.legend(
                    handles=[
                        Patch(facecolor="#2fc1ff", alpha=0.6, label="PASS"),
                        Patch(facecolor="#ff7b72", alpha=0.6, label="FAIL"),
                    ],
                    fontsize=8,
                )
                fig.tight_layout()
                charts["prob_box"] = fig_to_base64_png(fig)

    return charts


def build_saved_run_html(snapshot: dict, results: dict) -> str:
    run_name = str(snapshot.get("run_name", "Saved Run"))
    saved_at = str(snapshot.get("saved_at", ""))
    meta = results.get("meta", {}) or {}
    metrics = results.get("metrics", {}) or {}
    charts = build_saved_run_charts(results)
    class_report = results.get("classification_report", {}) or {}
    df_preview = results.get("dataframe", pd.DataFrame())
    if not isinstance(df_preview, pd.DataFrame):
        df_preview = pd.DataFrame()
    df_preview = df_preview.head(80)
    preview_html = (
        df_preview.to_html(index=False, border=0, classes="data-table")
        if not df_preview.empty
        else "<p>No table preview available.</p>"
    )

    metric_cards = []
    metric_keys = [
        ("Accuracy", "Accuracy"),
        ("Precision_weighted", "Precision (w)"),
        ("Recall_weighted", "Recall (w)"),
        ("F1_weighted", "F1 (w)"),
    ]
    for k, label in metric_keys:
        v = metrics.get(k, None)
        text = f"{float(v):.4f}" if isinstance(v, (int, float)) else "-"
        metric_cards.append(
            f"<div class='card metric'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(text)}</div></div>"
        )
    metric_cards.append(
        f"<div class='card metric'><div class='label'>Model</div><div class='value compact'>{html.escape(str(results.get('model_name', '-')))}</div></div>"
    )
    metric_cards.append(
        f"<div class='card metric'><div class='label'>Target</div><div class='value compact'>{html.escape(str(meta.get('target_column', '-')))}</div></div>"
    )
    metric_cards.append(
        f"<div class='card metric'><div class='label'>Classes</div><div class='value compact'>{html.escape(str(meta.get('num_classes', '-')))}</div></div>"
    )

    chart_blocks = []
    chart_labels = [
        ("distribution", "Class Distribution"),
        ("roc", "ROC Curves"),
        ("top_hist", "Top SHAP Histograms (PASS/FAIL)"),
        ("prob_box", "Probability by Group (PASS/FAIL)"),
    ]
    for key, label in chart_labels:
        if key in charts:
            chart_blocks.append(
                f"<div class='card chart'><h3>{html.escape(label)}</h3><img src='data:image/png;base64,{charts[key]}' alt='{html.escape(label)}' /></div>"
            )

    fi_sections = []
    shap_imp = results.get("shap_importance", {}) or {}
    if isinstance(shap_imp, dict):
        for cls_name, df_cls in shap_imp.items():
            if not isinstance(df_cls, pd.DataFrame) or df_cls.empty:
                continue
            fi_sections.append(
                f"<details class='card fi'><summary>{html.escape(str(cls_name))} - Top Features</summary><div class='table-wrap'>{df_cls.head(20).to_html(index=False, border=0, classes='data-table')}</div></details>"
            )

    per_class_rows = []
    class_order = []
    bins = results.get("bins", pd.DataFrame())
    if isinstance(bins, pd.DataFrame) and not bins.empty and "bin_name" in bins.columns:
        class_order = [str(x) for x in bins["bin_name"].tolist()]
    if not class_order and isinstance(class_report, dict):
        for k, v in class_report.items():
            if isinstance(v, dict) and {"precision", "recall", "f1-score"}.issubset(
                set(v.keys())
            ):
                class_order.append(str(k))
    for cls in class_order:
        row = class_report.get(cls, {})
        if not isinstance(row, dict):
            continue
        if not {"precision", "recall", "f1-score"}.issubset(set(row.keys())):
            continue
        per_class_rows.append(
            {
                "Class": cls,
                "Precision": f"{float(row.get('precision', 0.0)):.4f}",
                "Recall": f"{float(row.get('recall', 0.0)):.4f}",
                "F1": f"{float(row.get('f1-score', 0.0)):.4f}",
                "Support": int(row.get("support", 0)),
            }
        )
    per_class_html = (
        pd.DataFrame(per_class_rows).to_html(index=False, border=0, classes="data-table")
        if per_class_rows
        else "<p>No per-class metrics available.</p>"
    )

    meta_items = [
        ("Saved At", saved_at),
        ("Run Name", run_name),
        ("Mode", meta.get("mode", "")),
        ("Model Type", meta.get("model_type", "")),
        ("Data Path", meta.get("data_path", "")),
        ("Rows", meta.get("n_rows", "")),
        ("Columns (Original)", meta.get("n_cols_original", "")),
        ("Columns (After Filter)", meta.get("n_cols_after_filter", "")),
    ]
    meta_html = "".join(
        f"<div><span>{html.escape(str(k))}</span><strong>{html.escape(str(v))}</strong></div>"
        for k, v in meta_items
        if str(v).strip() != ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(run_name)} - Cisco Silicon Failure Characterization</title>
  <style>
    :root {{ --bg:#0f141c; --panel:#171f2b; --ink:#e7edf8; --muted:#9fb0c5; --cyan:#27bcff; --line:#2f3a4d; }}
    body {{ margin:0; font-family:"Avenir Next","Segoe UI",Arial,sans-serif; color:var(--ink); background:radial-gradient(circle at top right,#1d2f46,#0f141c 55%); }}
    .wrap {{ max-width:1320px; margin:28px auto; padding:0 18px 28px; }}
    h1 {{ font-size:30px; margin:6px 0 4px; }}
    .sub {{ color:var(--muted); margin-bottom:14px; }}
    .grid {{ display:grid; gap:14px; }}
    .metrics {{ grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
    .meta {{ grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }}
    .charts {{ grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }}
    .card {{ background:linear-gradient(180deg,#1b2432,#151d2a); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:0 10px 30px rgba(0,0,0,.25); }}
    .metric .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .metric .value {{ font-size:26px; font-weight:700; margin-top:6px; color:#edf5ff; line-height:1.15; overflow-wrap:anywhere; word-break:break-word; white-space:normal; }}
    .metric .value.compact {{ font-size:18px; line-height:1.25; }}
    .meta div {{ display:flex; justify-content:space-between; gap:10px; padding:6px 0; border-bottom:1px solid #253143; align-items:flex-start; }}
    .meta div:last-child {{ border-bottom:none; }}
    .meta span {{ color:var(--muted); }}
    .meta strong {{ max-width:70%; text-align:right; overflow-wrap:anywhere; word-break:break-word; }}
    .chart h3 {{ margin:2px 0 8px; color:#dce7f8; font-size:16px; }}
    .chart img {{ width:100%; border-radius:10px; background:#fff; }}
    .section {{ margin-top:14px; }}
    .table-wrap {{ width:100%; overflow-x:auto; overflow-y:hidden; border-radius:10px; border:1px solid #273549; }}
    .data-table {{ width:max-content; min-width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }}
    .data-table th,.data-table td {{ border:1px solid #304058; padding:6px 8px; text-align:left; }}
    .data-table th {{ background:#1f2b3c; position:sticky; top:0; }}
    details.fi summary {{ cursor:pointer; font-weight:600; color:#dbe8ff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(run_name)}</h1>
    <div class="sub">Saved run report - {html.escape(saved_at)}</div>

    <section class="grid metrics">
      {''.join(metric_cards)}
    </section>

    <section class="grid meta section card">
      {meta_html}
    </section>

    <section class="grid charts section">
      {''.join(chart_blocks) if chart_blocks else "<div class='card'>No chart images available in this snapshot.</div>"}
    </section>

    <section class="section card">
      <h3>Per-Class Metrics</h3>
      <div class="table-wrap">{per_class_html}</div>
    </section>

    <section class="section card">
      <h3>Data Preview</h3>
      <div class="table-wrap">{preview_html}</div>
    </section>

    <section class="section grid">
      {''.join(fi_sections) if fi_sections else "<div class='card'>No feature-importance tables available.</div>"}
    </section>
  </div>
</body>
</html>"""
