#!/usr/bin/env python3
"""
Correlación: tasa de cuentas vs. temperatura/presión.

Versión ajustada para que, cuando el archivo de conteos contiene columnas chNN,
la serie temporal de 4-fold reproduzca la misma definición usada en rolling.py:

- coincidencia 4-fold = exactamente un hit en cada uno de los 4 planos
- se construye primero la serie por segundo
- se rellenan con cero los segundos sin eventos
- al pasar a minuto se usan solo minutos completos

Si el archivo NO tiene canales chNN, el script conserva los modos previos
(rows / countcol / channels).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)


# ----------------- estilo -----------------
def set_article_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.dpi": 300,
    })


# ----------------- utilidades -----------------
def read_any_csv(path: Path) -> pd.DataFrame:
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


def find_col(cols: list[str], patterns: list[str]) -> str | None:
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None


def pick_col_by_substring(cols: list[str], keys: list[str]) -> str | None:
    for c in cols:
        cl = str(c).strip().lower()
        for k in keys:
            if k in cl:
                return c
    return None


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    den = np.sqrt(np.nansum(x * x)) * np.sqrt(np.nansum(y * y))
    return float(np.nansum(x * y) / den) if den != 0 else np.nan


def linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return float(a), float(b), float(r2)


def common_time_window(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    return max(a.min(), b.min()), min(a.max(), b.max())


def maybe_convert_temperature_to_c(temp: pd.Series) -> pd.Series:
    t = pd.to_numeric(temp, errors="coerce")
    med = float(np.nanmedian(t.to_numpy())) if t.notna().any() else np.nan
    if np.isfinite(med) and med > 150:
        return t - 273.15
    return t


def maybe_convert_pressure_to_hpa(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    med = float(np.nanmedian(p.to_numpy())) if p.notna().any() else np.nan
    name = getattr(p, "name", "") or ""
    name_l = str(name).lower()
    looks_pa = ("_pa" in name_l) or (" pa" in name_l) or (np.isfinite(med) and med > 2000.0)
    return (p / 100.0) if looks_pa else p


def parse_channels_from_header(cols: list[str]) -> dict[int, str]:
    num2name: dict[int, str] = {}
    for c in cols:
        m = CH_RE.match(str(c).strip())
        if m:
            num2name[int(m.group(1))] = c
    return num2name


def choose_channel_block(num2name: dict[int, str], channels_start: int = 1, n_channels: int = 60) -> list[str]:
    missing = [channels_start + k for k in range(n_channels) if (channels_start + k) not in num2name]
    if missing:
        raise ValueError(
            f"No encuentro bloque contiguo de {n_channels} canales empezando en ch{channels_start:02d}. "
            f"Faltan, por ejemplo: {missing[:10]}"
        )
    return [num2name[channels_start + k] for k in range(n_channels)]


def detectar_coincidencias_4fold(chunk: pd.DataFrame, ch_cols: list[str], n_bars: int) -> np.ndarray:
    if len(ch_cols) != 4 * n_bars:
        raise ValueError(
            f"n_bars={n_bars} inconsistente con {len(ch_cols)} canales. Se esperan 4*n_bars canales."
        )

    arr = chunk[ch_cols].to_numpy(dtype=np.int8)
    s1 = arr[:, 0:n_bars]
    s2 = arr[:, n_bars:2 * n_bars]
    s3 = arr[:, 2 * n_bars:3 * n_bars]
    s4 = arr[:, 3 * n_bars:4 * n_bars]

    c1 = s1.sum(axis=1)
    c2 = s2.sum(axis=1)
    c3 = s3.sum(axis=1)
    c4 = s4.sum(axis=1)

    return ((c1 == 1) & (c2 == 1) & (c3 == 1) & (c4 == 1)).astype(np.int8)


def _offset_to_timedelta(offset_alias: str) -> pd.Timedelta:
    off = pd.tseries.frequencies.to_offset(offset_alias)
    try:
        return pd.Timedelta(off)
    except Exception as exc:
        raise ValueError(
            f"La cadencia '{offset_alias}' no es una frecuencia fija. Usa algo como 1min, 5min, 30s, 1h."
        ) from exc


def _trim_to_full_bins(series_per_sec: pd.Series, dt: str) -> tuple[pd.Series, str]:
    if series_per_sec.empty:
        return series_per_sec.copy(), "sin datos"

    dt_td = _offset_to_timedelta(dt)
    first_ts = series_per_sec.index[0]
    last_ts = series_per_sec.index[-1]

    minute_all = series_per_sec.resample(dt).sum()

    first_full_bin = first_ts if first_ts.floor(dt) == first_ts else first_ts.ceil(dt)
    last_full_bin = (last_ts - (dt_td - pd.Timedelta(seconds=1))).floor(dt)

    if first_full_bin <= last_full_bin:
        out = minute_all.loc[first_full_bin:last_full_bin].copy()
        mode = "solo bins completos"
    else:
        out = minute_all.copy()
        mode = "bins parciales incluidos (duración total menor a un bin completo)"
    return out, mode


def build_count_label(dt: str) -> str:
    dt_l = dt.strip().lower()
    if dt_l in {"1min", "1t", "min", "1m"}:
        return "Cuentas/min"
    if dt_l in {"1s", "s", "sec", "1sec"}:
        return "Cuentas/s"
    return f"Cuentas por bin ({dt})"


# ----------------- sensor -----------------
def load_sensor(sensor_csv: Path, dt: str, rolling: str, tz: str | None = None) -> pd.DataFrame:
    df = read_any_csv(sensor_csv)
    cols = list(df.columns)

    tcol = find_col(cols, [r"^timestamp$", r"^time$", r"date", r"datetime"])
    if tcol is None:
        raise ValueError(f"[sensor] No encuentro columna temporal. Columnas: {cols[:30]}")

    tempcol = pick_col_by_substring(cols, ["temperatura", "temperature", "temp"])
    prescol = pick_col_by_substring(cols, ["presion", "pressure", "pres"])
    if tempcol is None or prescol is None:
        raise ValueError(
            f"[sensor] No detecto temperatura y/o presión. Detecté temp={tempcol}, pres={prescol}. "
            f"Columnas: {cols[:30]}"
        )

    ts = pd.to_datetime(df[tcol], errors="coerce")
    df = df.loc[ts.notna(), [tcol, tempcol, prescol]].copy()
    df[tcol] = ts.loc[ts.notna()].values

    if tz:
        if df[tcol].dt.tz is None:
            df[tcol] = df[tcol].dt.tz_localize(tz)
        else:
            df[tcol] = df[tcol].dt.tz_convert(tz)

    df = df.set_index(tcol).sort_index()

    out = pd.DataFrame(
        {
            "temperature_C": maybe_convert_temperature_to_c(df[tempcol]),
            "pressure_hPa": maybe_convert_pressure_to_hpa(df[prescol]),
        },
        index=df.index,
    )
    out = out.resample(dt).mean()
    out["temp_roll_med"] = out["temperature_C"].rolling(rolling, min_periods=1).median()
    out["pres_roll_med"] = out["pressure_hPa"].rolling(rolling, min_periods=1).median()
    return out


# ----------------- conteos -----------------
def load_counts_rate(
    counts_csv: Path,
    dt: str,
    counts_mode: str = "auto",
    tz: str | None = None,
    time_col_override: str | None = None,
    channels_start: int = 1,
    n_bars: int = 15,
) -> pd.DataFrame:
    header = pd.read_csv(counts_csv, nrows=0)
    cols = list(header.columns)

    tcol = time_col_override or find_col(cols, [r"^time$", r"^timestamp$", r"date", r"datetime"])
    if tcol is None:
        raise ValueError(f"[conteos] No encuentro columna temporal. Columnas: {cols[:30]}")

    countcol = pick_col_by_substring(cols, ["counts", "conteos", "conteo", "ncounts", "rate", "tasa"])
    num2name = parse_channels_from_header(cols)
    ch_cols_all = [c for c in cols if CH_RE.match(str(c).strip())]

    mode = counts_mode
    if mode == "auto":
        if countcol is not None:
            mode = "countcol"
        elif len(num2name) >= 4 * n_bars:
            mode = "4fold"
        elif len(ch_cols_all) > 0:
            mode = "rows"
        else:
            mode = "rows"

    if mode == "4fold":
        ch_cols = choose_channel_block(num2name, channels_start=channels_start, n_channels=4 * n_bars)
        usecols = [tcol] + ch_cols
        chunksize = 100_000

        coinc_per_sec_global: pd.Series | None = None
        t_start = None
        t_end = None

        for chunk in pd.read_csv(counts_csv, usecols=usecols, chunksize=chunksize, low_memory=True):
            ts = pd.to_datetime(chunk[tcol], errors="coerce")
            mask_t = ts.notna()
            if not mask_t.any():
                continue

            chunk = chunk.loc[mask_t].copy()
            ts = ts.loc[mask_t]
            if tz:
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize(tz)
                else:
                    ts = ts.dt.tz_convert(tz)

            t_sec = ts.dt.floor("s")
            tmin = t_sec.min()
            tmax = t_sec.max()
            if t_start is None or tmin < t_start:
                t_start = tmin
            if t_end is None or tmax > t_end:
                t_end = tmax

            coinc = detectar_coincidencias_4fold(chunk, ch_cols=ch_cols, n_bars=n_bars)
            if not np.any(coinc):
                continue

            df_tmp = pd.DataFrame({"time": t_sec.to_numpy(), "coinc4": coinc})
            grp = df_tmp.groupby("time", sort=False)["coinc4"].sum()
            coinc_per_sec_global = grp if coinc_per_sec_global is None else coinc_per_sec_global.add(grp, fill_value=0)

        if t_start is None or t_end is None:
            raise ValueError("[conteos] No pude leer tiempos válidos del archivo.")

        full_second_index = pd.date_range(start=t_start.floor("s"), end=t_end.floor("s"), freq="s")
        if coinc_per_sec_global is None:
            counts_per_sec = pd.Series(0, index=full_second_index, dtype=int, name="counts_per_sec")
        else:
            counts_per_sec = coinc_per_sec_global.sort_index().reindex(full_second_index, fill_value=0).astype(int)
            counts_per_sec.name = "counts_per_sec"

        counts_per_bin, bin_mode = _trim_to_full_bins(counts_per_sec, dt=dt)
        out = pd.DataFrame({
            "counts_per_min": counts_per_bin.astype(float),
            "n_obs": 1.0,
        })
        out.index.name = "time"
        out.attrs["counts_mode_used"] = "4fold"
        out.attrs["bin_mode"] = bin_mode
        return out

    chunksize = 300_000
    agg_sum: dict[pd.Timestamp, float] = {}
    agg_nobs: dict[pd.Timestamp, float] = {}

    usecols = [tcol]
    if mode == "countcol" and countcol is not None:
        usecols.append(countcol)
    elif mode == "channels" and len(ch_cols_all) > 0:
        usecols += ch_cols_all

    for chunk in pd.read_csv(counts_csv, usecols=usecols, chunksize=chunksize, low_memory=True):
        ts = pd.to_datetime(chunk[tcol], errors="coerce")
        mask = ts.notna()
        if not mask.any():
            continue

        ts = ts.loc[mask]
        if tz:
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(tz)
            else:
                ts = ts.dt.tz_convert(tz)

        tbin = ts.dt.floor(dt)

        if mode == "rows":
            gb = tbin.value_counts()
            for tb, c in gb.items():
                tb = pd.Timestamp(tb)
                agg_sum[tb] = agg_sum.get(tb, 0.0) + float(c)
                agg_nobs[tb] = agg_nobs.get(tb, 0.0) + float(c)

        elif mode == "countcol":
            if countcol is None:
                raise ValueError("[conteos] counts_mode=countcol pero no detecté columna de conteos.")
            v = pd.to_numeric(chunk.loc[mask, countcol], errors="coerce")
            tmp = pd.DataFrame({"t": tbin.to_numpy(), "v": v.to_numpy()})
            gb_sum = tmp.groupby("t", sort=False)["v"].sum(min_count=1)
            gb_n = tmp.groupby("t", sort=False)["v"].count()
            for tb in gb_n.index:
                n = float(gb_n.loc[tb])
                if n <= 0:
                    continue
                s = gb_sum.loc[tb]
                s = float(s) if pd.notna(s) else np.nan
                tb = pd.Timestamp(tb)
                agg_sum[tb] = agg_sum.get(tb, 0.0) + (0.0 if not np.isfinite(s) else s)
                agg_nobs[tb] = agg_nobs.get(tb, 0.0) + n

        elif mode == "channels":
            mat = chunk.loc[mask, ch_cols_all].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            v = mat.to_numpy().sum(axis=1)
            tmp = pd.DataFrame({"t": tbin.to_numpy(), "v": v})
            gb_sum = tmp.groupby("t", sort=False)["v"].sum()
            gb_n = tmp.groupby("t", sort=False)["v"].size()
            for tb in gb_n.index:
                n = float(gb_n.loc[tb])
                if n <= 0:
                    continue
                s = float(gb_sum.loc[tb])
                tb = pd.Timestamp(tb)
                agg_sum[tb] = agg_sum.get(tb, 0.0) + s
                agg_nobs[tb] = agg_nobs.get(tb, 0.0) + n
        else:
            raise ValueError(f"[conteos] counts_mode desconocido: {mode}")

    if not agg_sum:
        raise ValueError("[conteos] No pude agregar datos (¿columna tiempo inválida o archivo vacío?).")

    out = pd.DataFrame({
        "counts_per_min": pd.Series(agg_sum, dtype=float),
        "n_obs": pd.Series(agg_nobs, dtype=float),
    }).sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "time"
    out = out.loc[out["n_obs"] > 0]
    out.attrs["counts_mode_used"] = mode
    out.attrs["bin_mode"] = "solo donde existen datos"
    return out


# ----------------- plots -----------------
def plot_timeseries(df: pd.DataFrame, outbase: Path, title: str, y_label: str) -> None:
    set_article_style()
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 5.8), sharex=True, constrained_layout=True)

    axes[0].plot(df.index, df["counts_per_min"])
    axes[0].set_ylabel(y_label)

    axes[1].plot(df.index, df["temp_roll_med"])
    axes[1].set_ylabel("Temp (°C)\n(roll. med.)")

    axes[2].plot(df.index, df["pres_roll_med"])
    axes[2].set_ylabel("Pres (hPa)\n(roll. med.)")
    axes[2].set_xlabel("Tiempo")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)

    for ax in axes:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4)

    axes[0].set_title(title)
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"))
    plt.close(fig)


def plot_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, outbase: Path) -> dict:
    set_article_style()
    a, b, r2 = linfit(x, y)
    r = pearsonr(x, y)

    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]

    max_pts = 250_000
    if len(x2) > max_pts:
        step = max(1, len(x2) // max_pts)
        x2 = x2[::step]
        y2 = y2[::step]

    fig = plt.figure(figsize=(6.0, 4.2))
    plt.scatter(x2, y2, s=8, alpha=0.35, edgecolors="none")

    if np.isfinite(a) and np.isfinite(b):
        xx = np.linspace(np.nanmin(x2), np.nanmax(x2), 200)
        plt.plot(xx, a + b * xx, linewidth=1.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nPearson r = {r:.3f} | R² = {r2:.3f} | y = {a:.3g} + {b:.3g} x")
    plt.grid(True, which="major", linestyle="--", linewidth=0.6)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.4)
    fig.tight_layout()

    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"))
    plt.close(fig)
    return {"pearson_r": r, "intercept_a": a, "slope_b": b, "r2": r2, "n": int(mask.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cuentas vs temperatura/presión (rolling median), con ventana temporal común y correlaciones."
    )
    ap.add_argument("sensor_csv", type=Path, help="CSV del sensor (temperatura y presión).")
    ap.add_argument("counts_csv", type=Path, help="CSV de conteos (eventos o conteos).")
    ap.add_argument("--dt", default="10min", help="Cadencia común (default: 1min).")
    ap.add_argument("--rolling", default="15min", help="Ventana rolling (default: 10min).")
    ap.add_argument("--tz", default=None, help="Zona horaria IANA (ej: America/Bogota).")
    ap.add_argument(
        "--counts-mode",
        choices=["auto", "4fold", "rows", "countcol", "channels"],
        default="4fold",
        help=(
            "Cómo convertir el archivo de conteos. "
            "Si hay columnas chNN, auto usa 4fold para reproducir rolling.py."
        ),
    )
    ap.add_argument("--counts-time-col", default=None, help="Nombre columna tiempo en conteos (override).")
    ap.add_argument("--channels-start", type=int, default=1, help="Canal inicial del bloque contiguo chNN.")
    ap.add_argument("--n-bars", type=int, default=15, help="Número de barras por plano para 4-fold.")
    ap.add_argument("--outdir", default=None, help="Directorio de salida (opcional).")
    args = ap.parse_args()

    sensor_csv = args.sensor_csv.expanduser().resolve()
    counts_csv = args.counts_csv.expanduser().resolve()
    if not sensor_csv.exists():
        raise FileNotFoundError(sensor_csv)
    if not counts_csv.exists():
        raise FileNotFoundError(counts_csv)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (
        counts_csv.parent / f"corr_{counts_csv.stem}__{sensor_csv.stem}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    sensor = load_sensor(sensor_csv, dt=args.dt, rolling=args.rolling, tz=args.tz)
    counts = load_counts_rate(
        counts_csv,
        dt=args.dt,
        counts_mode=args.counts_mode,
        tz=args.tz,
        time_col_override=args.counts_time_col,
        channels_start=args.channels_start,
        n_bars=args.n_bars,
    )

    s_start, s_end = sensor.index.min(), sensor.index.max()
    c_start, c_end = counts.index.min(), counts.index.max()
    start, end = common_time_window(sensor.index, counts.index)
    if not (pd.notna(start) and pd.notna(end)) or start >= end:
        raise ValueError(
            "No hay traslape temporal entre archivos.\n"
            f"  sensor : [{s_start} , {s_end}]\n"
            f"  conteos: [{c_start} , {c_end}]"
        )

    sensor_c = sensor.loc[(sensor.index >= start) & (sensor.index <= end)]
    counts_c = counts.loc[(counts.index >= start) & (counts.index <= end)]

    df = counts_c.join(sensor_c[["temp_roll_med", "pres_roll_med"]], how="inner")
    df = df.dropna(subset=["temp_roll_med", "pres_roll_med", "counts_per_min"])

    xT = df["temp_roll_med"].to_numpy()
    xP = df["pres_roll_med"].to_numpy()
    y = df["counts_per_min"].to_numpy()
    rT = pearsonr(xT, y)
    rP = pearsonr(xP, y)

    y_label = build_count_label(args.dt)
    title = (
        f"Ventana común: {start} → {end} | dt={args.dt} | rolling={args.rolling}\n"
        f"counts_mode={counts.attrs.get('counts_mode_used')} | {counts.attrs.get('bin_mode')}\n"
        f"Pearson r: cuentas vs T = {rT:.3f} | cuentas vs P = {rP:.3f}"
    )
    plot_timeseries(df, outdir / "timeseries_counts_vs_env", title=title, y_label=y_label)

    res_T = plot_scatter(
        x=xT,
        y=y,
        xlabel="Temperatura (°C) (rolling median)",
        ylabel=y_label,
        title=f"{y_label} vs Temperatura",
        outbase=outdir / "scatter_counts_vs_temperature",
    )
    res_P = plot_scatter(
        x=xP,
        y=y,
        xlabel="Presión (hPa) (rolling median)",
        ylabel=y_label,
        title=f"{y_label} vs Presión",
        outbase=outdir / "scatter_counts_vs_pressure",
    )

    df.reset_index().to_csv(outdir / "fusion_counts_vs_env.csv", index=False)

    summary_lines = [
        f"sensor   : {sensor_csv}",
        f"conteos  : {counts_csv}",
        f"dt       : {args.dt}",
        f"rolling  : {args.rolling}",
        f"mode     : {counts.attrs.get('counts_mode_used')}",
        f"bins     : {counts.attrs.get('bin_mode')}",
        f"overlap  : {start} -> {end}",
        "",
        f"{y_label} vs Temperatura (rolling median):",
        f"  Pearson r : {res_T['pearson_r']:.6f}",
        f"  Fit       : y = {res_T['intercept_a']:.6g} + {res_T['slope_b']:.6g} x",
        f"  R^2       : {res_T['r2']:.6f}",
        "",
        f"{y_label} vs Presión (rolling median):",
        f"  Pearson r : {res_P['pearson_r']:.6f}",
        f"  Fit       : y = {res_P['intercept_a']:.6g} + {res_P['slope_b']:.6g} x",
        f"  R^2       : {res_P['r2']:.6f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("OK. Salida en:", outdir)
    print(f"  sensor   : [{s_start} , {s_end}]")
    print(f"  conteos  : [{c_start} , {c_end}]")
    print(f"  overlap  : [{start} , {end}]")
    print(f"  mode     : {counts.attrs.get('counts_mode_used')}")
    print(f"  bins     : {counts.attrs.get('bin_mode')}")
    print(f"  Pearson r ({y_label} vs T): {rT:.4f}")
    print(f"  Pearson r ({y_label} vs P): {rP:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
