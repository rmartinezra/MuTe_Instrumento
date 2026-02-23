#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Global plotting style for "paper-like" figures ---
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

try:
    from tqdm import tqdm  # noqa: F401
except Exception:
    tqdm = None

CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)


def _try_import_polars():
    try:
        import polars as pl  # noqa: F401
        return pl
    except Exception:
        return None


def _to_polars_duration(dt: str) -> str:
    """Convert pandas-like offsets ('1min', '10min') to Polars durations ('1m', '10m')."""
    dt = dt.strip()
    if dt.endswith("min"):
        return dt[:-3] + "m"
    return dt


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Simple ranking (average ranks for ties)."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def pearsonr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    return float((x * y).sum() / den) if den != 0 else np.nan


def spearmanr(x, y) -> float:
    rx = _rankdata(np.asarray(x))
    ry = _rankdata(np.asarray(y))
    return pearsonr(rx, ry)


def parse_channels_from_header(cols: list[str]) -> dict[int, str]:
    num2name = {}
    for c in cols:
        m = CH_RE.match(str(c).strip())
        if m:
            n = int(m.group(1))
            num2name.setdefault(n, c)
    return num2name


def choose_channel_block(num2name: dict[int, str], channels_start: int) -> list[str]:
    if all((channels_start + k) in num2name for k in range(60)):
        n = 60
    elif all((channels_start + k) in num2name for k in range(64)):
        n = 64
    else:
        missing = [
            channels_start + k
            for k in range(60)
            if (channels_start + k) not in num2name
        ][:20]
        raise ValueError(
            f"Cannot find a contiguous block from ch{channels_start}. "
            f"Missing (examples): {missing}"
        )
    return [num2name[channels_start + k] for k in range(n)]


def load_flow_timeseries(
    flow_csv: Path,
    time_col: str,
    channels_start: int,
    dt: str,
    days: list[int],
    area: float,
    backend: str = "auto",
):
    # Read only header to detect channels
    header_cols = list(pd.read_csv(flow_csv, nrows=0).columns)
    num2name = parse_channels_from_header(header_cols)
    ch_cols = choose_channel_block(num2name, channels_start)

    pl_mod = _try_import_polars()
    if backend == "auto":
        backend = "polars" if pl_mod is not None else "pandas"
    if backend == "polars" and pl_mod is None:
        backend = "pandas"

    # ---------- POLARS BACKEND (streaming) ----------
    if backend == "polars":
        import polars as pl

        try:
            lf = pl.scan_csv(str(flow_csv), try_parse_dates=True)
        except TypeError:
            lf = pl.scan_csv(str(flow_csv))

        lf = lf.select([time_col] + ch_cols)

        lf = lf.with_columns(
            pl.col(time_col).cast(pl.Datetime, strict=False).alias(time_col)
        ).drop_nulls([time_col])

        lf = lf.filter(pl.col(time_col).dt.day().is_in(days))

        dt_pol = _to_polars_duration(dt)
        lf = lf.with_columns(pl.col(time_col).dt.truncate(dt_pol).alias("t"))

        lf = lf.with_columns(
            pl.sum_horizontal([pl.col(c) for c in ch_cols]).alias("counts")
        )

        per = (
            lf.group_by("t")
            .agg(pl.col("counts").sum().alias("counts"))
            .sort("t")
        )

        sel = per.select(["t", "counts"])
        try:
            out = sel.collect(engine="streaming")
        except TypeError:
            out = sel.collect()

        t = out["t"].to_numpy()
        counts = out["counts"].to_numpy()

        s = pd.Series(counts, index=pd.to_datetime(t)).sort_index()
        flux = s / area
        flux.name = "flux"
        return flux

    # ---------- PANDAS BACKEND (chunked) ----------
    agg_counts: dict[pd.Timestamp, float] = {}
    chunksize = 200_000

    for chunk in pd.read_csv(
        flow_csv,
        usecols=[time_col] + ch_cols,
        engine="c",
        low_memory=True,
        chunksize=chunksize,
    ):
        t = pd.to_datetime(chunk[time_col], errors="coerce")
        mask_valid = t.notna()
        if not mask_valid.any():
            continue
        chunk = chunk.loc[mask_valid].copy()
        t = t.loc[mask_valid]

        mask_days = t.dt.day.isin(days)
        if not mask_days.any():
            continue
        chunk = chunk.loc[mask_days].copy()
        t = t.loc[mask_days]

        chunk[time_col] = t
        chunk["t"] = t.dt.floor(dt)

        chunk["counts"] = chunk[ch_cols].to_numpy(copy=False).sum(axis=1)

        gb = chunk.groupby("t", sort=False)["counts"].sum()

        for tb, c in gb.items():
            agg_counts[tb] = agg_counts.get(tb, 0.0) + float(c)

    if not agg_counts:
        raise ValueError("No data found for the selected days in the flow CSV.")

    per = pd.Series(agg_counts)
    per.index = pd.to_datetime(per.index)
    per = per.sort_index()
    flux = (per / area).rename("flux")
    return flux


def read_any_csv(path: Path) -> pd.DataFrame:
    """Lightweight separator detection using only the first line."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
    except Exception:
        header = ""

    candidates = [",", ";", "\t", "|"]
    counts = {sep: header.count(sep) for sep in candidates}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = ","

    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def _pick_col_by_substring(cols: list[str], keys: list[str]) -> str | None:
    """Return first column whose lowercase name contains any key substring."""
    for c in cols:
        cl = str(c).strip().lower()
        for k in keys:
            if k in cl:
                return c
    return None


def load_sensor_timeseries(
    sensor_paths: list[Path],
    timestamp_col: str,
    dt: str,
    days: list[int],
):
    df = pd.concat([read_any_csv(p) for p in sensor_paths], ignore_index=True)

    if timestamp_col not in df.columns:
        raise ValueError(
            f"Sensor CSV does not contain timestamp column '{timestamp_col}'. "
            f"Columns found: {list(df.columns)[:30]}"
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df[df[timestamp_col].dt.day.isin(days)].set_index(timestamp_col)

    # Detect columns by substring (handles temperatura_C, presion_Pa, altura_m, etc.)
    cols = list(df.columns)

    col_temp = _pick_col_by_substring(cols, ["temperatura", "temperature", "temp"])
    col_pres = _pick_col_by_substring(cols, ["presion", "pressure", "pres"])
    col_alt  = _pick_col_by_substring(cols, ["altura", "altitude", "height"])
    col_irr  = _pick_col_by_substring(cols, ["irradiancia", "irradiance", "radiacion", "radiation"])

    keep = [c for c in [col_temp, col_pres, col_alt, col_irr] if c is not None]
    if not keep:
        raise ValueError(
            "No se detectaron columnas ambientales. Esperaba algo como "
            "temperatura*, presion*, altura* o irradiancia*."
        )

    # Convert to numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pressure unit handling: if looks like Pa, convert to hPa
    pressure_unit = None
    if col_pres is not None:
        pres = df[col_pres]
        pres_med = float(np.nanmedian(pres.to_numpy())) if pres.notna().any() else np.nan
        name_l = str(col_pres).lower()
        looks_pa = ("_pa" in name_l) or (" pa" in name_l) or (pres_med > 2000.0)
        if looks_pa:
            df[col_pres] = df[col_pres] / 100.0
            pressure_unit = "hPa (converted from Pa)"
        else:
            pressure_unit = "hPa"

    # Resample
    df = df[keep].resample(dt).mean()

    # Rename to canonical names
    rename = {}
    if col_temp:
        rename[col_temp] = "temperature"
    if col_pres:
        rename[col_pres] = "pressure"
    if col_alt:
        rename[col_alt] = "altitude"
    if col_irr:
        rename[col_irr] = "irradiance"
    df = df.rename(columns=rename)

    # store meta as attribute (optional)
    df.attrs["pressure_unit"] = pressure_unit

    return df


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Rolling muon flux + environmental sensor (T/P/Alt/I) and correlation analysis."
        )
    )
    ap.add_argument("flow_csv", help="Flow CSV (time + ch* columns)")
    ap.add_argument(
        "--sensor",
        nargs="+",
        required=True,
        help="Sensor CSV file(s) (timestamp + variables). You can pass several files.",
    )
    ap.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=[16, 17, 18, 19, 20],
        help="Day-of-month values to select (default: 16 17 18 19 20).",
    )
    ap.add_argument("--time-col", default="time", help="Time column in flow CSV.")
    ap.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column in sensor CSV.",
    )
    ap.add_argument(
        "--channels-start",
        type=int,
        default=1,
        help="1 for ch1..ch60; 0 for ch00..ch59.",
    )
    ap.add_argument(
        "--dt",
        default="1min",
        help="Common cadence (default: 1min). Examples: 1s, 10s, 1min.",
    )
    ap.add_argument(
        "--rolling",
        default="10min",
        help="Rolling window (default: 10min). Examples: 5min, 30min.",
    )
    ap.add_argument(
        "--area",
        type=float,
        default=0.36,
        help="Detector area in m² for flux calculation (default: 0.36).",
    )
    ap.add_argument(
        "--backend",
        choices=["auto", "polars", "pandas"],
        default="auto",
        help="Backend for the flow file. 'polars' uses multi-core streaming.",
    )
    ap.add_argument(
        "--outdir",
        default="",
        help="Output directory (default: flow_dir/fusion_<flow_stem>).",
    )
    args = ap.parse_args()

    flow_csv = Path(args.flow_csv).expanduser().resolve()
    sensor_paths = [Path(p).expanduser().resolve() for p in args.sensor]
    if not flow_csv.exists():
        raise FileNotFoundError(flow_csv)
    for p in sensor_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else (flow_csv.parent / f"fusion_{flow_csv.stem}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Muon flux
    flux = load_flow_timeseries(
        flow_csv,
        args.time_col,
        args.channels_start,
        args.dt,
        args.days,
        args.area,
        backend=args.backend,
    )

    flux_roll = flux.rolling(args.rolling, min_periods=1).median().rename("flux_roll")

    # 2) Environmental sensor
    sensor = load_sensor_timeseries(sensor_paths, args.timestamp_col, args.dt, args.days)
    sensor_roll = sensor.rolling(args.rolling, min_periods=1).mean()

    # 3) Merge
    df = pd.concat([flux_roll, sensor_roll], axis=1).dropna(subset=["flux_roll"])

    # -------- Time-series plot --------
    max_points_ts = 800_000
    plot_df = df
    if len(df) > max_points_ts:
        step = max(1, len(df) // max_points_ts)
        plot_df = df.iloc[::step]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_flux = "black"
    color_temp = "tab:red"
    color_pres = "tab:green"
    color_alt = "tab:blue"
    color_irr = "tab:orange"

    ax1.plot(
        plot_df.index,
        plot_df["flux_roll"],
        linewidth=1.8,
        color=color_flux,
        label="Muon flux (rolling median)",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Muon flux (counts / bin / m²)")
    ax1.tick_params(axis="both", labelsize=12)

    axes = [ax1]
    offset = 60

    # pressure label depends on conversion
    punit = sensor.attrs.get("pressure_unit") or "hPa"
    pres_label = f"Pressure ({'hPa' if 'hPa' in punit else punit})"

    for name, ylab, color in [
        ("temperature", "Temperature (°C)", color_temp),
        ("pressure", pres_label, color_pres),
        ("altitude", "Altitude (m)", color_alt),
        ("irradiance", "Irradiance (W/m²)", color_irr),
    ]:
        if name in plot_df.columns:
            ax = ax1.twinx()
            ax.spines["right"].set_position(("outward", offset))
            offset += 60
            ax.plot(
                plot_df.index,
                plot_df[name],
                linewidth=1.4,
                linestyle="--",
                color=color,
                label=f"{ylab} (rolling mean)",
            )
            ax.set_ylabel(ylab)
            ax.tick_params(axis="y", labelsize=12)
            axes.append(ax)

    lines, labs = [], []
    for ax in axes:
        l, la = ax.get_legend_handles_labels()
        lines += l
        labs += la
    if lines:
        ax1.legend(lines, labs, loc="upper left", frameon=True)

    ax1.set_title(
        f"Rolling muon flux and environmental variables "
        f"(days {args.days}, dt={args.dt}, window={args.rolling})"
    )
    ax1.grid(True, alpha=0.3, linestyle=":")

    out_main = out_dir / "rolling_flux_env.png"
    fig.tight_layout()
    fig.savefig(out_main, dpi=300)
    plt.close(fig)

    # -------- Scatter + correlations --------
    def scatter(var: str, label: str, color: str, filename: str):
        if var not in df.columns:
            return
        d = df[["flux_roll", var]].dropna()
        if len(d) < 10:
            return

        r_p = pearsonr(d["flux_roll"].to_numpy(), d[var].to_numpy())
        r_s = spearmanr(d["flux_roll"].to_numpy(), d[var].to_numpy())

        max_points_scatter = 300_000
        d_plot = d
        if len(d) > max_points_scatter:
            step = max(1, len(d) // max_points_scatter)
            d_plot = d.iloc[::step]

        fig = plt.figure(figsize=(7, 5.5))
        plt.scatter(
            d_plot[var],
            d_plot["flux_roll"],
            s=10,
            alpha=0.5,
            color=color,
            edgecolors="none",
        )
        plt.xlabel(label, fontsize=14)
        plt.ylabel("Muon flux (rolling median)", fontsize=14)
        plt.title(
            f"Muon flux vs {label}\nPearson = {r_p:.3f}, Spearman = {r_s:.3f}",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)

    scatter("temperature", "Temperature (°C)", color_temp, "scatter_flux_vs_temperature.png")
    scatter("pressure", pres_label, color_pres, "scatter_flux_vs_pressure.png")
    scatter("altitude", "Altitude (m)", color_alt, "scatter_flux_vs_altitude.png")
    scatter("irradiance", "Irradiance (W/m²)", color_irr, "scatter_flux_vs_irradiance.png")

    # Save merged dataset
    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(
        out_dir / "fusion_rolling.csv", index=False
    )

    print("Done. Outputs written to:", out_dir)
    print(" -", out_main)
    for f in [
        "scatter_flux_vs_temperature.png",
        "scatter_flux_vs_pressure.png",
        "scatter_flux_vs_altitude.png",
        "scatter_flux_vs_irradiance.png",
    ]:
        p = out_dir / f
        if p.exists():
            print(" -", p)


if __name__ == "__main__":
    raise SystemExit(main())
