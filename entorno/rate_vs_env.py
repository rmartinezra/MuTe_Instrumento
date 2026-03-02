#!/usr/bin/env python3
"""
Correlación: tasa de cuentas por minuto vs. temperatura/presión (rolling median).

Uso mínimo (solo dos archivos):
  python rate_vs_env.py sensor.csv conteos.csv

El script:
1) Lee ambos CSV y detecta columnas de tiempo y de variables (T/P) de forma automática.
2) Agrega conteos a "cuentas por minuto" (por defecto cuenta filas por minuto).
3) Remuestrea sensor a 1 minuto y calcula rolling median (por defecto 10 min).
4) Recorta al intervalo temporal común.
5) Grafica series temporales + gráficos de dispersión (scatter) y calcula correlación lineal.

Salida: crea un directorio con figuras + CSV fusionado + resumen de correlaciones.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CH_RE = re.compile(r"^ch\d+$", re.IGNORECASE)


# ----------------- estilo "artículo" -----------------
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
    """Detección ligera de separador usando solo la primera línea."""
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
    """Ajuste y = a + b x; devuelve (a, b, R2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)  # y = b x + a
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return float(a), float(b), float(r2)


def common_time_window(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    a0, a1 = a.min(), a.max()
    b0, b1 = b.min(), b.max()
    start = max(a0, b0)
    end = min(a1, b1)
    return start, end


def maybe_convert_temperature_to_c(temp: pd.Series) -> pd.DataFrame:
    t = pd.to_numeric(temp, errors="coerce")
    med = float(np.nanmedian(t.to_numpy())) if t.notna().any() else np.nan
    if np.isfinite(med) and med > 150:  # Kelvin
        return t - 273.15
    return t


def maybe_convert_pressure_to_hpa(p: pd.Series) -> pd.DataFrame:
    p = pd.to_numeric(p, errors="coerce")
    med = float(np.nanmedian(p.to_numpy())) if p.notna().any() else np.nan
    name = getattr(p, "name", "") or ""
    name_l = str(name).lower()
    looks_pa = ("_pa" in name_l) or (" pa" in name_l) or (np.isfinite(med) and med > 2000.0)
    return (p / 100.0) if looks_pa else p


# ----------------- carga sensor -----------------
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
            f"[sensor] No detecto temperatura y/o presión. "
            f"Detecté temp={tempcol}, pres={prescol}. Columnas: {cols[:30]}"
        )

    ts = pd.to_datetime(df[tcol], errors="coerce")
    df = df.loc[ts.notna(), [tcol, tempcol, prescol]].copy()
    df[tcol] = ts.loc[ts.notna()].values

    if tz:
        # Si no hay tz, localiza. Si hay, convierte.
        if df[tcol].dt.tz is None:
            df[tcol] = df[tcol].dt.tz_localize(tz)
        else:
            df[tcol] = df[tcol].dt.tz_convert(tz)

    df = df.set_index(tcol).sort_index()

    temp = maybe_convert_temperature_to_c(df[tempcol])
    pres = maybe_convert_pressure_to_hpa(df[prescol])

    out = pd.DataFrame({"temperature_C": temp, "pressure_hPa": pres}, index=df.index)

    # remuestreo a dt
    out = out.resample(dt).mean()

    # rolling median
    out["temp_roll_med"] = out["temperature_C"].rolling(rolling, min_periods=1).median()
    out["pres_roll_med"] = out["pressure_hPa"].rolling(rolling, min_periods=1).median()
    return out


# ----------------- carga conteos -----------------
def load_counts_rate(
    counts_csv: Path,
    dt: str,
    counts_mode: str = "auto",
    tz: str | None = None,
    time_col_override: str | None = None,
) -> pd.DataFrame:
    """
    Devuelve DataFrame indexado por minuto con:
      - counts_per_min : cuentas por minuto
      - n_obs          : número de observaciones que contribuyeron al bin (para saber si "existe" ese minuto)

    counts_mode:
      - auto: heurística
      - rows: tasa = número de filas por minuto (eventos)
      - countcol: usa una columna tipo counts/conteos y suma por minuto
      - channels: suma columnas ch* por fila y luego suma por minuto
    """
    # leer solo header para detectar columnas
    header = pd.read_csv(counts_csv, nrows=0)
    cols = list(header.columns)

    tcol = time_col_override or find_col(cols, [r"^time$", r"^timestamp$", r"date", r"datetime"])
    if tcol is None:
        raise ValueError(f"[conteos] No encuentro columna temporal. Columnas: {cols[:30]}")

    countcol = pick_col_by_substring(cols, ["counts", "conteos", "conteo", "ncounts", "rate", "tasa"])
    ch_cols = [c for c in cols if CH_RE.match(str(c).strip())]

    mode = counts_mode
    if mode == "auto":
        if countcol is not None:
            mode = "countcol"
        elif len(ch_cols) >= 20:
            mode = "rows"  # por defecto: eventos -> filas
        else:
            mode = "rows"

    chunksize = 300_000
    agg_sum: dict[pd.Timestamp, float] = {}
    agg_nobs: dict[pd.Timestamp, float] = {}

    usecols = [tcol]
    if mode == "countcol" and countcol is not None:
        usecols.append(countcol)
    elif mode == "channels" and len(ch_cols) > 0:
        usecols += ch_cols

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
            v = pd.to_numeric(chunk.loc[mask, countcol], errors="coerce")  # mantiene NaN si falta
            tmp = pd.DataFrame({"t": tbin.to_numpy(), "v": v.to_numpy()})

            # suma (si todo es NaN en el minuto -> NaN) y número de observaciones válidas
            gb_sum = tmp.groupby("t", sort=False)["v"].sum(min_count=1)
            gb_n   = tmp.groupby("t", sort=False)["v"].count()

            for tb in gb_n.index:
                n = float(gb_n.loc[tb])
                if n <= 0:
                    continue  # este minuto no "existe" en el archivo de conteos (no hay datos válidos)
                s = gb_sum.loc[tb]
                s = float(s) if pd.notna(s) else np.nan
                tb = pd.Timestamp(tb)
                agg_sum[tb] = agg_sum.get(tb, 0.0) + (0.0 if not np.isfinite(s) else s)
                agg_nobs[tb] = agg_nobs.get(tb, 0.0) + n

        elif mode == "channels":
            if len(ch_cols) == 0:
                raise ValueError("[conteos] counts_mode=channels pero no hay columnas ch*.")

            mat = chunk.loc[mask, ch_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            v = mat.to_numpy().sum(axis=1)

            tmp = pd.DataFrame({"t": tbin.to_numpy(), "v": v})
            gb_sum = tmp.groupby("t", sort=False)["v"].sum()
            gb_n   = tmp.groupby("t", sort=False)["v"].size()

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
    out = out.loc[out["n_obs"] > 0]  # minutos donde realmente hubo datos en el archivo de conteos
    return out



# ----------------- plots -----------------
def plot_timeseries(df: pd.DataFrame, outbase: Path, title: str) -> None:
    set_article_style()

    fig, axes = plt.subplots(3, 1, figsize=(6.7, 5.8), sharex=True, constrained_layout=True)

    axes[0].plot(df.index, df["counts_per_min"])
    axes[0].set_ylabel("Cuentas/min")

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

    a, b, r2 = linfit(x, y)  # y = a + b x
    r = pearsonr(x, y)

    mask = np.isfinite(x) & np.isfinite(y)
    x2 = x[mask]
    y2 = y[mask]

    # downsample si es enorme
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

    return {"pearson_r": r, "intercept_a": a, "slope_b": b, "r2": r2, "n": int((np.isfinite(x) & np.isfinite(y)).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cuentas/min vs temperatura/presión (rolling median), con ventana temporal común y correlaciones."
    )
    ap.add_argument("sensor_csv", type=Path, help="CSV del sensor (temperatura y presión).")
    ap.add_argument("counts_csv", type=Path, help="CSV de conteos (eventos o conteos).")
    ap.add_argument("--dt", default="1min", help="Cadencia común (default: 1min).")
    ap.add_argument("--rolling", default="10min", help="Ventana rolling (default: 10min).")
    ap.add_argument("--tz", default=None, help="Zona horaria IANA (ej: America/Bogota).")
    ap.add_argument(
        "--counts-mode",
        choices=["auto", "rows", "countcol", "channels"],
        default="auto",
        help="Cómo convertir el archivo de conteos a cuentas/min (default: auto).",
    )
    ap.add_argument("--counts-time-col", default=None, help="Nombre columna tiempo en conteos (override).")
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

    # 1) series
    sensor = load_sensor(sensor_csv, dt=args.dt, rolling=args.rolling, tz=args.tz)
    counts = load_counts_rate(
        counts_csv,
        dt=args.dt,
        counts_mode=args.counts_mode,
        tz=args.tz,
        time_col_override=args.counts_time_col,
    )

    # 2) ventana común
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

    # 3) fusion
    # Usamos los minutos del archivo de conteos como referencia ("solo donde los conteos existen")
    df = counts_c.join(sensor_c[["temp_roll_med", "pres_roll_med"]], how="inner")
    df = df.dropna(subset=["temp_roll_med", "pres_roll_med", "counts_per_min"])

    # 4) correlaciones
    xT = df["temp_roll_med"].to_numpy()
    xP = df["pres_roll_med"].to_numpy()
    y  = df["counts_per_min"].to_numpy()
    rT = pearsonr(xT, y)
    rP = pearsonr(xP, y)

    # 5) plots
    title = (
        f"Ventana común: {start} → {end} | dt={args.dt} | rolling={args.rolling}\n"
        f"Pearson r: cuentas vs T = {rT:.3f} | cuentas vs P = {rP:.3f}"
    )
    plot_timeseries(df, outdir / "timeseries_counts_vs_env", title=title)

    res_T = plot_scatter(
        x=df["temp_roll_med"].to_numpy(),
        y=df["counts_per_min"].to_numpy(),
        xlabel="Temperatura (°C) (rolling median)",
        ylabel="Cuentas/min",
        title="Cuentas/min vs Temperatura",
        outbase=outdir / "scatter_counts_vs_temperature",
    )
    res_P = plot_scatter(
        x=df["pres_roll_med"].to_numpy(),
        y=df["counts_per_min"].to_numpy(),
        xlabel="Presión (hPa) (rolling median)",
        ylabel="Cuentas/min",
        title="Cuentas/min vs Presión",
        outbase=outdir / "scatter_counts_vs_pressure",
    )

    # 6) guardar dataset + resumen
    df.reset_index().to_csv(outdir / "fusion_1min_rolling.csv", index=False)

    summary_lines = [
        f"sensor   : {sensor_csv}",
        f"conteos  : {counts_csv}",
        f"dt       : {args.dt}",
        f"rolling  : {args.rolling}",
        f"overlap  : {start} -> {end}",
        "",
        "Cuentas/min vs Temperatura (rolling median):",
        f"  Pearson r : {res_T['pearson_r']:.6f}",
        f"  Fit       : y = {res_T['intercept_a']:.6g} + {res_T['slope_b']:.6g} x",
        f"  R^2       : {res_T['r2']:.6f}",
        "",
        "Cuentas/min vs Presión (rolling median):",
        f"  Pearson r : {res_P['pearson_r']:.6f}",
        f"  Fit       : y = {res_P['intercept_a']:.6g} + {res_P['slope_b']:.6g} x",
        f"  R^2       : {res_P['r2']:.6f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("OK. Salida en:", outdir)
    print(f"  sensor   : [{s_start} , {s_end}]")
    print(f"  conteos  : [{c_start} , {c_end}]")
    print(f"  overlap  : [{start} , {end}]")
    print(f"  Pearson r (cuentas vs T): {rT:.4f}")
    print(f"  Pearson r (cuentas vs P): {rP:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
