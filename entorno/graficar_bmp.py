#!/usr/bin/env python3
"""
Graficar temperatura y presión vs. tiempo (estilo artículo) desde un CSV.

Ejemplo:
  python plot_pres_temp.py archivo_concatenado.csv -o presion_temperatura.pdf

Requisitos:
  pip install pandas matplotlib
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _find_col(cols: list[str], patterns: list[str]) -> str | None:
    """Devuelve el primer nombre de columna que coincida con alguno de los patrones (regex)."""
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _maybe_convert_temperature_to_c(temp: pd.Series) -> pd.Series:
    """Convierte a °C si parece estar en Kelvin (mediana > 150)."""
    temp_num = _to_numeric(temp)
    med = np.nanmedian(temp_num.to_numpy())
    if np.isfinite(med) and med > 150:
        return temp_num - 273.15
    return temp_num


def _maybe_convert_pressure_to_hpa(p: pd.Series) -> pd.Series:
    """Convierte a hPa si parece estar en Pa (mediana > 2000)."""
    p_num = _to_numeric(p)
    med = np.nanmedian(p_num.to_numpy())
    if np.isfinite(med) and med > 2000:
        return p_num / 100.0
    return p_num


def load_pt_csv(path: Path, tz: str | None = None) -> tuple[pd.DataFrame, str, str, str]:
    df = pd.read_csv(path)
    cols = list(df.columns)

    tcol = _find_col(cols, [r"^timestamp$", r"date", r"datetime", r"time"])
    if tcol is None:
        raise ValueError("No encuentro una columna de tiempo. Busqué: timestamp/date/datetime/time.")

    tempcol = _find_col(cols, [r"temp", r"temperatura", r"^t(_|$)"])
    if tempcol is None:
        raise ValueError("No encuentro una columna de temperatura (temp/temperatura).")

    pcol = _find_col(cols, [r"pres", r"pressure", r"^p(_|$)"])
    if pcol is None:
        raise ValueError("No encuentro una columna de presión (pres/presion/pressure).")

    ts = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    if ts.isna().all():
        raise ValueError(f"No pude convertir la columna '{tcol}' a fechas.")

    if tz:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz)
        else:
            ts = ts.dt.tz_convert(tz)

    temp_c = _maybe_convert_temperature_to_c(df[tempcol])
    pres_hpa = _maybe_convert_pressure_to_hpa(df[pcol])

    out = pd.DataFrame({"time": ts, "temp": temp_c, "pres": pres_hpa}).dropna(subset=["time"])
    out = out.sort_values("time")
    return out, tcol, tempcol, pcol


def set_article_style():
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


def plot_pt(df: pd.DataFrame, outpath: Path, title: str | None = None,
            width_in: float = 6.7, height_in: float = 4.2):
    set_article_style()

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=True,
        figsize=(width_in, height_in),
        constrained_layout=True,
    )

    ax1.plot(df["time"], df["temp"])
    ax1.set_ylabel("Temperatura (°C)")
    if title:
        ax1.set_title(title)

    ax2.plot(df["time"], df["pres"])
    ax2.set_ylabel("Presión (hPa)")
    ax2.set_xlabel("Tiempo")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    for ax in (ax1, ax2):
        ax.grid(True, which="major", linestyle="--", linewidth=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Grafica temperatura y presión vs. tiempo (estilo artículo).")
    ap.add_argument("csv", type=Path, help="CSV de entrada.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Archivo de salida (.pdf recomendado, también .png).")
    ap.add_argument("--tz", type=str, default=None,
                    help="Zona horaria IANA (ej: America/Bogota).")
    ap.add_argument("--title", type=str, default=None, help="Título opcional.")
    ap.add_argument("--width-in", type=float, default=6.7, help="Ancho en pulgadas.")
    ap.add_argument("--height-in", type=float, default=4.2, help="Alto en pulgadas.")
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = args.csv.with_suffix("")
        out = out.with_name(out.name + "_presion_temperatura.pdf")

    df, tcol, tempcol, pcol = load_pt_csv(args.csv, tz=args.tz)
    plot_pt(df, out, title=args.title, width_in=args.width_in, height_in=args.height_in)

    print(f"OK: {out}")
    print(f"Columnas usadas -> tiempo: {tcol} | temp: {tempcol} | presión: {pcol}")


if __name__ == "__main__":
    main()
