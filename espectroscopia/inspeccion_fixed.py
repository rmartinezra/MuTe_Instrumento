#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: robust CSV loading
# -----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Try reading CSV with a couple of common comment styles.
    """
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


def pick_col(cols, candidates):
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    return None


def run_duration_seconds(df: pd.DataFrame, time_col: str | None) -> float | None:
    if not time_col or time_col not in df.columns:
        return None
    v = df[time_col].to_numpy()
    if len(v) < 2:
        return None
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None

    # heuristic: detect unit by column name
    name = time_col.lower()
    if "us" in name:
        return (vmax - vmin) / 1e6
    if "ns" in name:
        return (vmax - vmin) / 1e9
    if "ms" in name:
        return (vmax - vmin) / 1e3
    # default assume seconds
    return float(vmax - vmin)


def multiplicity_per_trigger(df: pd.DataFrame, event_col: str, mult_col: str | None) -> pd.Series:
    """
    Return multiplicity per trigger. Prefer Num_Chs if present; otherwise count rows per trigger.
    """
    if mult_col and mult_col in df.columns:
        return df.groupby(event_col)[mult_col].first()
    # fallback: number of hits (rows) per trigger
    return df.groupby(event_col).size().rename("Num_Chs")


def saturation_fraction_per_channel(df: pd.DataFrame, ch_col: str, pha_col: str, adc_max: int) -> pd.Series:
    """
    Fraction of hits at ADC max per channel.
    """
    g = df.groupby(ch_col)[pha_col].apply(lambda x: float(np.mean(x.to_numpy() >= adc_max)))
    # keep channels sorted if numeric
    try:
        g = g.sort_index()
    except Exception:
        pass
    return g


def triggers_with_any_saturation(df: pd.DataFrame, event_col: str, pha_col: str, adc_max: int) -> float:
    """
    Fraction of triggers that have at least one saturated hit.
    """
    per_trg_max = df.groupby(event_col)[pha_col].max()
    return float(np.mean(per_trg_max.to_numpy() >= adc_max))


# -----------------------------
# Plotting per file
# -----------------------------
def analyze_file(
    csv_path: str,
    outdir: str | None,
    adc_max: int = 8191,
    bins_mult_max: int = 64,
    bins_pha: int = 200,
):
    df = read_csv_robust(csv_path)
    cols = list(df.columns)

    # Try to infer standard columns
    event_col = pick_col(cols, ["Trg_Id", "event_id", "TrgId", "trigger_id", "TriggerId"])
    ch_col = pick_col(cols, ["CH_Id", "channel", "ch", "Channel"])
    pha_col = pick_col(cols, ["PHA_HG", "PHA", "PHAHG", "EnergyHG"])
    mult_col = pick_col(cols, ["Num_Chs", "n_channels", "nch", "Multiplicity"])
    time_col = pick_col(cols, ["TStamp_us", "timestamp", "TStamp_ns", "time_us", "time_ns", "TimeStamp_us"])

    if event_col is None or ch_col is None or pha_col is None:
        raise RuntimeError(
            f"[ERROR] Missing required columns in {os.path.basename(csv_path)}.\n"
            f"Found columns: {cols}\n"
            f"Need event_col (e.g. Trg_Id), ch_col (e.g. CH_Id), pha_col (e.g. PHA_HG)."
        )

    # Basic arrays
    pha = df[pha_col].to_numpy()
    mult = multiplicity_per_trigger(df, event_col=event_col, mult_col=mult_col)
    dur_s = run_duration_seconds(df, time_col=time_col)

    # Metrics
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

    # Print metrics to terminal
    fn = os.path.basename(csv_path)
    print("\n" + "=" * 80)
    print(f"File: {fn}")
    print(f"Rows (hits): {len(df):d}")
    print(f"Triggers: {n_trg:d}")
    if dur_s is not None:
        print(f"Duration: {dur_s:.3f} s")
    if rate_hz is not None:
        print(f"Trigger rate: {rate_hz:.3f} Hz")

    print("\nMultiplicity (Num_Chs per trigger):")
    print(f"  median = {med_mult:.1f}")
    print(f"  p90    = {p90_mult:.1f}")
    print(f"  frac(Num_Chs=4)  = {100.0*frac_mult_eq4:.2f}%")
    print(f"  frac(Num_Chs>20) = {100.0*frac_mult_gt20:.2f}%")

    print("\nPHA (HG):")
    print(f"  min(PHA)  = {min_pha:d}   <-- useful to verify effective ZS in HIGH mode")
    print(f"  p01/p50/p99 = {p01_pha:.1f} / {p50_pha:.1f} / {p99_pha:.1f}")
    print(f"  ADC max used for saturation = {adc_max:d}")
    print(f"  frac(hits saturated)        = {100.0*frac_hits_sat:.2f}%")
    print(f"  frac(triggers w/ any sat)   = {100.0*frac_trg_any_sat:.2f}%")

    # Prepare output paths
    outdir = outdir or str(Path(csv_path).resolve().parent)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(fn)[0]
    fig_path = os.path.join(outdir, f"{base}__summary.png")

    # -----------------------------
    # Plot: single figure, 3 panels
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Multiplicity with log Y
    ax = axes[0]
    bins = np.arange(-0.5, bins_mult_max + 0.5, 1)
    ax.hist(mult.to_numpy(), bins=bins, histtype="stepfilled", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("# active channels per trigger (Num_Chs)")
    ax.set_ylabel("Triggers (log scale)")
    ax.set_title("Multiplicity (log Y)")
    ax.grid(True, which="both", alpha=0.2)

    # (2) PHA distribution (log Y)
    ax = axes[1]
    # Focus bins from low percentile to ADC max, but include max spike if present
    lo = max(0, int(np.quantile(pha, 0.001)))
    hi = max(int(np.nanmax(pha)), adc_max)
    if hi <= lo:
        hi = lo + 1
    bins = np.linspace(lo, hi, bins_pha)
    ax.hist(pha, bins=bins, histtype="step", linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel(f"{pha_col} [ADC]")
    ax.set_ylabel("Hits (log scale)")
    ax.set_title("PHA distribution (HG, log Y)")
    ax.grid(True, which="both", alpha=0.2)

    # (3) Saturation fraction per channel
    ax = axes[2]
    ch = sat_by_ch.index.to_numpy()
    frac = sat_by_ch.to_numpy()
    ax.bar(ch, frac)
    ax.set_xlabel("Channel ID")
    ax.set_ylabel(f"Fraction of hits at ADC max ({adc_max})")
    ax.set_title("Saturation fraction per channel")
    ax.set_ylim(0, max(0.05, float(np.nanmax(frac)) * 1.15))
    ax.grid(True, axis="y", alpha=0.2)

    # Suptitle with key metrics
    subtitle = (
        f"{fn}\n"
        f"Triggers={n_trg}"
        + (f", Rate={rate_hz:.2f} Hz" if rate_hz is not None else "")
        + f", sat(hits)={100*frac_hits_sat:.2f}%, sat(triggers)={100*frac_trg_any_sat:.2f}%"
        + f", frac(NumChs=4)={100*frac_mult_eq4:.2f}%"
    )
    fig.suptitle(subtitle, fontsize=11)

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved figure: {fig_path}")
    return fig_path


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Inspect CAEN/Janus CSV files: multiplicity, PHA_HG distribution, saturation per channel."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Input directory or CSV file. Default: current directory.",
    )
    ap.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern if 'path' is a directory. Default: *.csv",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory for PNG figures. Default: same folder as each CSV.",
    )
    ap.add_argument(
        "--adc-max",
        type=int,
        default=8191,
        help="ADC max value used to tag saturation. Default: 8191",
    )
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file():
        files = [str(p)]
    else:
        # directory mode
        d = str(p) if p.exists() else "."
        files = sorted(glob.glob(os.path.join(d, args.pattern)))

    if not files:
        print(f"[ERROR] No files found in '{args.path}' with pattern '{args.pattern}'.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        try:
            analyze_file(f, outdir=args.outdir, adc_max=args.adc_max)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)}: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
