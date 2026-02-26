#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comparador.py — Compare multiple CAEN/Janus CSV files in ONE figure (3 panels)
and additionally save a NEW figure with a table-like summary of metrics.

Panels (compare figure):
  (1) Multiplicity distribution (Num_Chs per trigger), log Y, step overlays
  (2) PHA_HG distribution, log Y, step overlays
  (3) Saturation fraction per channel (PHA_HG == ADCmax), line overlays

Extra figure:
  Metrics summary table (one row per metric, one column per file/label),
  including Trigger rate (Hz).
"""

import argparse
import os
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: robust CSV loading
# -----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    for comment in (None, "#", "/"):
        try:
            if comment is None:
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, comment=comment)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
    raise RuntimeError(f"Could not read CSV: {path}")


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def run_duration_seconds(df: pd.DataFrame, time_col: Optional[str]) -> Optional[float]:
    if not time_col or time_col not in df.columns:
        return None

    v = df[time_col].to_numpy()
    if len(v) < 2:
        return None

    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None

    name = time_col.lower()
    if "us" in name:
        return float((vmax - vmin) / 1e6)
    if "ns" in name:
        return float((vmax - vmin) / 1e9)
    if "ms" in name:
        return float((vmax - vmin) / 1e3)
    return float(vmax - vmin)


def multiplicity_per_trigger(df: pd.DataFrame, event_col: str, mult_col: Optional[str]) -> pd.Series:
    if mult_col and mult_col in df.columns:
        return df.groupby(event_col)[mult_col].first()
    return df.groupby(event_col).size().rename("Num_Chs")


def saturation_fraction_per_channel(df: pd.DataFrame, ch_col: str, pha_col: str, adc_max: int) -> pd.Series:
    g = df.groupby(ch_col)[pha_col].apply(lambda x: float(np.mean(x.to_numpy() >= adc_max)))
    try:
        g = g.sort_index()
    except Exception:
        pass
    return g


def triggers_with_any_saturation(df: pd.DataFrame, event_col: str, pha_col: str, adc_max: int) -> float:
    per_trg_max = df.groupby(event_col)[pha_col].max()
    return float(np.mean(per_trg_max.to_numpy() >= adc_max))


def compute_metrics(
    df: pd.DataFrame,
    event_col: str,
    ch_col: str,
    pha_col: str,
    mult_col: Optional[str],
    time_col: Optional[str],
    adc_max: int,
) -> Dict[str, Any]:
    pha = df[pha_col].to_numpy()
    mult = multiplicity_per_trigger(df, event_col=event_col, mult_col=mult_col)
    dur_s = run_duration_seconds(df, time_col=time_col)

    n_trg = int(mult.shape[0])
    rate_hz = (n_trg / dur_s) if (dur_s and dur_s > 0) else None

    frac_hits_sat = float(np.mean(pha >= adc_max))
    frac_trg_any_sat = triggers_with_any_saturation(df, event_col, pha_col, adc_max)

    med_mult = float(np.median(mult.to_numpy()))
    p90_mult = float(np.quantile(mult.to_numpy(), 0.90))
    frac_mult_eq4 = float(np.mean(mult.to_numpy() == 4))
    frac_mult_gt20 = float(np.mean(mult.to_numpy() > 20))

    min_pha = int(np.nanmin(pha))
    p01_pha = float(np.quantile(pha, 0.01))
    p50_pha = float(np.quantile(pha, 0.50))
    p99_pha = float(np.quantile(pha, 0.99))

    sat_by_ch = saturation_fraction_per_channel(df, ch_col, pha_col, adc_max)

    return {
        "n_hits": int(len(df)),
        "n_triggers": n_trg,
        "duration_s": float(dur_s) if dur_s is not None else None,
        "rate_hz": float(rate_hz) if rate_hz is not None else None,
        "median_mult": med_mult,
        "p90_mult": p90_mult,
        "frac_mult_eq4": frac_mult_eq4,
        "frac_mult_gt20": frac_mult_gt20,
        "min_pha": min_pha,
        "p01_pha": p01_pha,
        "p50_pha": p50_pha,
        "p99_pha": p99_pha,
        "frac_hits_sat": frac_hits_sat,
        "frac_trg_any_sat": frac_trg_any_sat,
        "sat_by_ch": sat_by_ch,
        "mult_series": mult,
        "pha_array": pha,
    }


def wrap_label(s: str, width: int = 32) -> str:
    # Mantiene el texto completo pero lo “dobla” para que quepa en la tabla
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def fmt(v: Any, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    if kind == "int":
        return f"{int(v)}"
    if kind == "float1":
        return f"{float(v):.1f}"
    if kind == "float2":
        return f"{float(v):.2f}"
    if kind == "float3":
        return f"{float(v):.3f}"
    if kind == "pct":
        return f"{100.0*float(v):.2f}%"
    return str(v)


def save_metrics_table_figure(datasets: List[Dict[str, Any]], outpath: str):
    """
    Figura nueva con tabla:
      Filas = métricas (nombre completo)
      Columnas = archivos/labels
    """
    rows: List[Tuple[str, str, str]] = [
        ("Hits (rows)", "n_hits", "int"),
        ("Triggers (events)", "n_triggers", "int"),
        ("Duration (s)", "duration_s", "float2"),
        ("Trigger rate (Hz)", "rate_hz", "float3"),
        ("Multiplicity median (#channels)", "median_mult", "float1"),
        ("Multiplicity p90 (#channels)", "p90_mult", "float1"),
        ("Fraction Num_Chs = 4", "frac_mult_eq4", "pct"),
        ("Fraction Num_Chs > 20", "frac_mult_gt20", "pct"),
        ("min(PHA_HG) [ADC]", "min_pha", "int"),
        ("PHA_HG p50 [ADC]", "p50_pha", "float1"),
        ("PHA_HG p99 [ADC]", "p99_pha", "float1"),
        ("Saturated hits (PHA=ADCmax)", "frac_hits_sat", "pct"),
        ("Triggers with >=1 saturated hit", "frac_trg_any_sat", "pct"),
    ]

    col_labels = [wrap_label(d["label"], width=34) for d in datasets]
    row_labels = [r[0] for r in rows]

    cell_text: List[List[str]] = []
    for (_name, key, kind) in rows:
        row = []
        for d in datasets:
            m = d["metrics"]
            row.append(fmt(m.get(key, None), kind))
        cell_text.append(row)

    # Tamaño dinámico según #archivos
    ncols = len(col_labels)
    width = max(10, 2.6 * ncols)
    height = 0.55 * len(rows) + 1.6

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    ax.set_title("Metrics summary (per file)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compare multiple CAEN/Janus CSV files in one figure + save a metrics-summary figure."
    )
    ap.add_argument("files", nargs="+", help="CSV files to compare (space-separated).")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional labels (same count as files). Default: basename.")
    ap.add_argument("--outdir", default=".", help="Output directory for PNGs.")
    ap.add_argument("--name", default="comparison", help="Base name for output PNGs.")
    ap.add_argument("--adc-max", type=int, default=8191, help="ADC max value used to tag saturation. Default: 8191")
    ap.add_argument("--mult-max", type=int, default=64, help="Max multiplicity bin shown. Default: 64")
    ap.add_argument("--pha-bins", type=int, default=250, help="Number of bins for PHA histogram. Default: 250")
    ap.add_argument("--pha-min", type=int, default=None,
                    help="Force PHA histogram min (ADC). Default: auto from data.")
    ap.add_argument("--pha-max", type=int, default=None,
                    help="Force PHA histogram max (ADC). Default: auto from data (at least adc-max).")
    args = ap.parse_args()

    files = [str(Path(f)) for f in args.files]
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f"File not found: {f}")

    labels = args.labels
    if labels is None or len(labels) == 0:
        labels = [os.path.basename(f) for f in files]
    else:
        if len(labels) != len(files):
            raise ValueError("If you pass --labels, you must provide exactly one label per file.")

    os.makedirs(args.outdir, exist_ok=True)

    datasets: List[Dict[str, Any]] = []
    for f, lab in zip(files, labels):
        df = read_csv_robust(f)
        cols = list(df.columns)

        event_col = pick_col(cols, ["Trg_Id", "event_id", "TrgId", "trigger_id", "TriggerId"])
        ch_col = pick_col(cols, ["CH_Id", "channel", "ch", "Channel"])
        pha_col = pick_col(cols, ["PHA_HG", "PHA", "PHAHG", "EnergyHG"])
        mult_col = pick_col(cols, ["Num_Chs", "n_channels", "nch", "Multiplicity"])
        time_col = pick_col(cols, ["TStamp_us", "timestamp", "TStamp_ns", "time_us", "time_ns", "TimeStamp_us"])

        if event_col is None or ch_col is None or pha_col is None:
            raise RuntimeError(
                f"[ERROR] Missing required columns in {os.path.basename(f)}.\n"
                f"Found columns: {cols}\n"
                f"Need event_col (e.g. Trg_Id), ch_col (e.g. CH_Id), pha_col (e.g. PHA_HG)."
            )

        metrics = compute_metrics(
            df,
            event_col=event_col, ch_col=ch_col, pha_col=pha_col,
            mult_col=mult_col, time_col=time_col,
            adc_max=args.adc_max
        )
        datasets.append({"file": f, "label": lab, "metrics": metrics})

        # Terminal print (incluye Hz)
        m = metrics
        print("\n" + "=" * 80)
        print(f"File: {os.path.basename(f)}")
        print(f"Label: {lab}")
        print(f"Rows (hits): {m['n_hits']}")
        print(f"Triggers: {m['n_triggers']}")
        if m["duration_s"] is not None:
            print(f"Duration: {m['duration_s']:.3f} s")
        if m["rate_hz"] is not None:
            print(f"Trigger rate (Hz): {m['rate_hz']:.3f}")

        print("\nMultiplicity (Num_Chs per trigger):")
        print(f"  median = {m['median_mult']:.1f}")
        print(f"  p90    = {m['p90_mult']:.1f}")
        print(f"  Fraction Num_Chs = 4  = {100.0*m['frac_mult_eq4']:.2f}%")
        print(f"  Fraction Num_Chs > 20 = {100.0*m['frac_mult_gt20']:.2f}%")

        print("\nPHA (HG):")
        print(f"  min(PHA)  = {m['min_pha']}   <-- useful to verify effective ZS in HIGH mode")
        print(f"  p01/p50/p99 = {m['p01_pha']:.1f} / {m['p50_pha']:.1f} / {m['p99_pha']:.1f}")
        print(f"  ADC max used for saturation = {args.adc_max}")
        print(f"  Saturated hits               = {100.0*m['frac_hits_sat']:.2f}%")
        print(f"  Triggers with any saturation = {100.0*m['frac_trg_any_sat']:.2f}%")

    # Common PHA histogram limits
    all_pha = np.concatenate([d["metrics"]["pha_array"] for d in datasets])
    pha_min = args.pha_min if args.pha_min is not None else int(np.quantile(all_pha, 0.001))
    pha_max = args.pha_max if args.pha_max is not None else int(max(np.max(all_pha), args.adc_max))
    if pha_max <= pha_min:
        pha_max = pha_min + 1

    # Build comparison figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    # Panel 1: multiplicity (log Y), step overlays
    ax = axes[0]
    bins_mult = np.arange(-0.5, args.mult_max + 0.5, 1)
    for d in datasets:
        mult = d["metrics"]["mult_series"].to_numpy()
        ax.hist(mult, bins=bins_mult, histtype="step", linewidth=2, label=d["label"])
    ax.set_yscale("log")
    ax.set_xlabel("# active channels per trigger (Num_Chs)")
    ax.set_ylabel("Triggers (log scale)")
    ax.set_title("Multiplicity (log Y)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)

    # Panel 2: PHA distribution (log Y), step overlays
    ax = axes[1]
    bins_pha = np.linspace(pha_min, pha_max, args.pha_bins)
    for d in datasets:
        pha = d["metrics"]["pha_array"]
        ax.hist(pha, bins=bins_pha, histtype="step", linewidth=2, label=d["label"])
    ax.set_yscale("log")
    ax.set_xlabel("PHA_HG [ADC]")
    ax.set_ylabel("Hits (log scale)")
    ax.set_title("PHA distribution (HG, log Y)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)

    # Panel 3: saturation per channel (overlay lines)
    ax = axes[2]
    ch_grid = np.arange(0, 64, 1)
    for d in datasets:
        s = d["metrics"]["sat_by_ch"]
        s_full = s.reindex(ch_grid).fillna(0.0).to_numpy()
        ax.plot(ch_grid, s_full, marker="o", linewidth=1.8, label=d["label"])
    ax.set_xlabel("Channel ID")
    ax.set_ylabel(f"Fraction of hits at ADC max ({args.adc_max})")
    ax.set_title("Saturation fraction per channel")
    ymax = float(np.max([d["metrics"]["sat_by_ch"].max() for d in datasets]))
    ax.set_ylim(0, max(0.05, ymax * 1.15))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8)

    # Suptitle with short summary (includes Hz)
    summary_bits = []
    for d in datasets:
        m = d["metrics"]
        hz = "N/A" if m["rate_hz"] is None else f"{m['rate_hz']:.2f} Hz"
        summary_bits.append(
            f"{d['label']}: {hz}, satHits={100*m['frac_hits_sat']:.2f}%, "
            f"satTrg={100*m['frac_trg_any_sat']:.2f}%, frac4={100*m['frac_mult_eq4']:.1f}%"
        )
    fig.suptitle(" | ".join(summary_bits), fontsize=9)

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    out_compare = os.path.join(args.outdir, f"{args.name}__compare.png")
    fig.savefig(out_compare, dpi=220)
    plt.close(fig)

    # NEW: metrics summary figure
    out_metrics = os.path.join(args.outdir, f"{args.name}__metrics.png")
    save_metrics_table_figure(datasets, out_metrics)

    print("\n" + "=" * 80)
    print(f"Saved comparison figure: {out_compare}")
    print(f"Saved metrics figure:    {out_metrics}")
    print("Done.")


if __name__ == "__main__":
    main()
