"""
Utility functions to export K+‐clearance results.
"""

import os, datetime, json
from pathlib import Path

import pandas as pd
from pyqtgraph.exporters import ImageExporter   # bundled with pyqtgraph

def results_to_csv(results: list[dict], out_dir: str | Path) -> Path:
    """
    Serialise the list of per‑segment dicts *results* into a tidy CSV.

    Returns the absolute path of the file created.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"results_{ts}.csv"

    # pick the scalar fields you want in the spreadsheet
    df = pd.DataFrame([{
        "Segment":       r.get("peak"),
        "V₀ (baseline)": r.get("v_baseline"),
        "Vₚ (peak)":     r.get("v_peak"),
        "ΔV (amplitude)": r.get("delta_v"),
        "V₉₀ (90%)":     r.get("v_baseline") + 0.9 * r.get("delta_v", 0),
        "V₁₀ (10%)":     r.get("v_baseline") + 0.1 * r.get("delta_v", 0),
        "[K⁺]₀ (baseline)": r.get("k_baseline"),
        "[K⁺]ₚ (peak)":     r.get("k_peak"),
        "Δ[K⁺] (amplitude)": r.get("delta_k"),
        "V₉₀ (mM)":      r.get("k90"),
        "V₁₀ (mM)":      r.get("k10"),
        "τ (Tau)":       r.get("tau"),
        "T3":            r.get("T3"),
        "λ = 1/τ":       r.get("lambda"),
        "tw (weighted)": r.get("tau_weighted"),
        "λ𝚠 (weighted)": r.get("lambda_weighted"),
        "Decay 90→10%":  r.get("decay_time"),
        "R² (fit)":      r.get("r2"),
    } for r in results])

    df.set_index('peak').T.to_csv(csv_path)
    return csv_path


def export_clearance_plots(plot_area, out_dir: str | Path, image_fmt="png", selected_names=None):

    """
    Save every *segment* tab in IxDPlotArea as <prefix>_seg<N>.<fmt>

    *plot_area*  – instance of IxDPlotArea  
    *image_fmt*  – 'png', 'jpg', 'pdf', or 'svg' (exporters support these)

    Returns list of Path objects for the files created.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    created = []
    # Tab index 0 is the raw sweep; segments start at 1
    for idx in range(1, plot_area.tabs.count()):
        widget = plot_area.tabs.widget(idx)          # PlotWidget
        exporter = ImageExporter(widget.plotItem)    # grab the plotItem
        fname = out_dir / f"clearance_seg{idx}.{image_fmt}"
        exporter.export(str(fname))
        created.append(fname)

    return created
