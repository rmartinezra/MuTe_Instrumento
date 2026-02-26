#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import pandas as pd

# Área efectiva [m^2]
AREA_DEFAULT = 0.36

# Regex genérico para columnas tipo ch1, ch01, ch60, etc.
CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)


def parse_channels_from_header(cols):
    """Mapa número -> nombre de columna para columnas chXX."""
    num2name = {}
    for c in cols:
        m = CH_RE.match(str(c).strip())
        if m:
            n = int(m.group(1))
            num2name.setdefault(n, c)
    return num2name


def choose_channel_block(num2name, channels_start=1, n_channels=60):
    """Devuelve la lista de nombres de columnas ch{start}..ch{start+n-1}."""
    missing = [channels_start + k for k in range(n_channels)
               if (channels_start + k) not in num2name]
    if missing:
        raise ValueError(
            f"No encuentro bloque contiguo de {n_channels} canales empezando en ch{channels_start}. "
            f"Faltan (ejemplo): {missing[:10]}"
        )
    return [num2name[channels_start + k] for k in range(n_channels)]


def construir_histograma(csv_path: Path,
                         time_col: str = "time",
                         channels_start: int = 1,
                         area: float = AREA_DEFAULT,
                         chunk_size: int = 100_000):
    # --- Preparar canales ---
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    num2name = parse_channels_from_header(header)
    ch_cols = choose_channel_block(num2name, channels_start=channels_start, n_channels=60)

    half = len(ch_cols) // 2
    panel1_cols = ch_cols[:half]
    panel2_cols = ch_cols[half:]

    # Series globales (conteos/s) por segundo
    panel1_global = None
    panel2_global = None

    usecols = [time_col] + ch_cols
    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunk_size,
    )

    if tqdm is not None:
        iterator = tqdm(reader, desc="Leyendo y agregando", unit="chunk", dynamic_ncols=True)
    else:
        iterator = reader

    for chunk in iterator:
        # Parsear y truncar tiempo a segundo
        t = pd.to_datetime(chunk[time_col], errors="coerce")
        m = t.notna()
        if not m.any():
            continue
        chunk = chunk.loc[m]
        t = t.dt.floor("S")
        chunk[time_col] = t

        # Agrupar por segundo y sumar canales
        grp = chunk.groupby(time_col, sort=False)[ch_cols].sum()

        s1 = grp[panel1_cols].sum(axis=1)
        s2 = grp[panel2_cols].sum(axis=1)

        if panel1_global is None:
            panel1_global = s1
            panel2_global = s2
        else:
            panel1_global = panel1_global.add(s1, fill_value=0.0)
            panel2_global = panel2_global.add(s2, fill_value=0.0)

    if panel1_global is None or panel2_global is None:
        raise RuntimeError("No se encontraron datos válidos tras el parseo de tiempo.")

    # Filtro básico y normalización por área
    mask1 = panel1_global.between(0, 3600)
    mask2 = panel2_global.between(0, 3600)

    rate1 = (panel1_global[mask1] / area).to_numpy(dtype=float)
    rate2 = (panel2_global[mask2] / area).to_numpy(dtype=float)

    N1, N2 = rate1.size, rate2.size

    mean1 = float(rate1.mean()) if N1 > 0 else float("nan")
    std1 = float(rate1.std(ddof=1)) if N1 > 1 else float("nan")
    err1 = std1 / np.sqrt(N1) if N1 > 0 and np.isfinite(std1) else float("nan")

    mean2 = float(rate2.mean()) if N2 > 0 else float("nan")
    std2 = float(rate2.std(ddof=1)) if N2 > 1 else float("nan")
    err2 = std2 / np.sqrt(N2) if N2 > 0 and np.isfinite(std2) else float("nan")

    print("Panel 1 (canales 1–30):")
    print(f"  N    = {N1}")
    print(f"  mean = {mean1:.6f} cnt/s/m^2")
    print(f"  std  = {std1:.6f} cnt/s/m^2")
    print(f"  err  = {err1:.6f} cnt/s/m^2")
    print("Panel 2 (canales 31–60):")
    print(f"  N    = {N2}")
    print(f"  mean = {mean2:.6f} cnt/s/m^2")
    print(f"  std  = {std2:.6f} cnt/s/m^2")
    print(f"  err  = {err2:.6f} cnt/s/m^2")

    # --- Figura con estética tipo revista ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 100,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = 50

    # Histogramas normalizados
    ax.hist(
        rate1,
        bins=bins,
        density=True,
        alpha=0.45,
        edgecolor="black",
        linewidth=0.8,
        label="Panel 1 (ch1–ch30)",
    )
    ax.hist(
        rate2,
        bins=bins,
        density=True,
        alpha=0.45,
        edgecolor="black",
        linewidth=0.8,
        label="Panel 2 (ch31–ch60)",
    )

    # Líneas de media y ±1σ
    ax.axvline(mean1, color="C0", linestyle="-", linewidth=2)
    ax.axvline(mean1 - std1, color="C0", linestyle="--", linewidth=1)
    ax.axvline(mean1 + std1, color="C0", linestyle="--", linewidth=1)

    ax.axvline(mean2, color="C1", linestyle="-", linewidth=2)
    ax.axvline(mean2 - std2, color="C1", linestyle="--", linewidth=1)
    ax.axvline(mean2 + std2, color="C1", linestyle="--", linewidth=1)

    ax.set_title("Histogram of count rate per second / m² (Panels 1 & 2)")
    ax.set_xlabel(r"Rate [counts s$^{-1}$ m$^{-2}$]")
    ax.set_ylabel("Normalized frequency")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1)

    # Leyenda centrada arriba para no pelearse con los cuadros
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    # Dos cuadros separados, uno a la izquierda y otro a la derecha
    text1 = (
        "Panel 1\n"
        rf"$\mu = {mean1:.3f} \pm {err1:.3f}$" + "\n"
        rf"$\sigma = {std1:.3f}$" + "\n"
        rf"$N = {N1:d}$"
    )
    text2 = (
        "Panel 2\n"
        rf"$\mu = {mean2:.3f} \pm {err2:.3f}$" + "\n"
        rf"$\sigma = {std2:.3f}$" + "\n"
        rf"$N = {N2:d}$"
    )

    ax.text(
        0.02,
        0.98,
        text1,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    ax.text(
        0.98,
        0.98,
        text2,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()

    # Carpeta "graficas_<archivo>" junto al CSV, como en rollingV1
    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "histograma_paneles.png"

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfica guardada en: {out_path}")


def main():
    pa = argparse.ArgumentParser(description="Histograma de tasa por panel (1–60) con estadísticos.")
    pa.add_argument("archivo", help="Archivo CSV de entrada")
    pa.add_argument("--time-col", default="time", help="Nombre de la columna de tiempo (por defecto: 'time')")
    pa.add_argument("--area", type=float, default=AREA_DEFAULT, help="Área efectiva [m^2] (defecto: 0.36)")
    pa.add_argument("--channels-start", type=int, default=1,
                    help="Primer canal del bloque continuo (defecto: 1 => ch1..ch60)")
    pa.add_argument("--chunk-size", type=int, default=100_000,
                    help="Tamaño de chunk para lectura por partes (defecto: 100000 filas)")

    args = pa.parse_args()
    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    construir_histograma(
        csv_path,
        time_col=args.time_col,
        channels_start=args.channels_start,
        area=args.area,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
