#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # si no está instalado, simplemente no hay barra de progreso
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Convierte coincidencias 4-panel (time,ch00..ch63) "
            "en un mapa de píxeles discretos (Δx,Δy) usando canales 1–60."
        )
    )
    p.add_argument("archivo", help="CSV de entrada (time,ch00..ch63)")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Filas por chunk para procesar en streaming (defecto: 200000).",
    )
    p.add_argument(
        "--max-delta",
        type=int,
        default=14,
        help="Máximo |Δ| en barras (defecto: 14 para 15 barras por plano).",
    )
    p.add_argument(
        "--pitch-cm",
        type=float,
        default=4.0,
        help="Pitch (ancho de barra) en cm (defecto: 4.0 cm).",
    )
    p.add_argument(
        "--distance-cm",
        type=float,
        default=70.0,
        help="Distancia entre paneles en cm (defecto: 70.0 cm).",
    )
    return p.parse_args()


def build_pixel_map(csv_path: Path, chunk_size: int, max_delta: int):
    """
    Lee el CSV por chunks y acumula un mapa N(Δx,Δy) usando SOLO canales 1..60,
    asumidos como:
      - Plano X superior: ch01..ch15
      - Plano Y superior: ch16..ch30
      - Plano X inferior: ch31..ch45
      - Plano Y inferior: ch46..ch60

    Solo se usan eventos donde en cada plano hay EXACTAMENTE un canal encendido.
    """
    side = 2 * max_delta + 1
    counts = np.zeros((side, side), dtype=np.int64)

    ch_cols = [f"ch{i:02d}" for i in range(64)]
    usecols = ["time"] + ch_cols

    g1_idx = np.arange(1, 16)   # ch01..ch15  (X top)
    g2_idx = np.arange(16, 31)  # ch16..ch30  (Y top)
    g3_idx = np.arange(31, 46)  # ch31..ch45  (X bottom)
    g4_idx = np.arange(46, 61)  # ch46..ch60  (Y bottom)

    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunk_size,
    )

    if tqdm is not None:
        iterator = tqdm(reader, desc="Procesando eventos", unit="chunk", dynamic_ncols=True)
    else:
        iterator = reader

    total_rows = 0
    used_events = 0

    for chunk in iterator:
        total_rows += len(chunk)

        arr = chunk[ch_cols].to_numpy(dtype=np.int8)

        s1 = arr[:, g1_idx]
        s2 = arr[:, g2_idx]
        s3 = arr[:, g3_idx]
        s4 = arr[:, g4_idx]

        c1 = s1.sum(axis=1)
        c2 = s2.sum(axis=1)
        c3 = s3.sum(axis=1)
        c4 = s4.sum(axis=1)

        mask = (c1 == 1) & (c2 == 1) & (c3 == 1) & (c4 == 1)
        if not np.any(mask):
            continue

        idx1_local = s1.argmax(axis=1)
        idx2_local = s2.argmax(axis=1)
        idx3_local = s3.argmax(axis=1)
        idx4_local = s4.argmax(axis=1)

        idx1_valid = idx1_local[mask]
        idx2_valid = idx2_local[mask]
        idx3_valid = idx3_local[mask]
        idx4_valid = idx4_local[mask]

        dx = idx3_valid - idx1_valid
        dy = idx4_valid - idx2_valid

        valid_delta = (
            (dx >= -max_delta) & (dx <= max_delta) &
            (dy >= -max_delta) & (dy <= max_delta)
        )
        if not np.any(valid_delta):
            continue

        dx = dx[valid_delta]
        dy = dy[valid_delta]

        used_events += dx.size

        ix = dx + max_delta
        iy = dy + max_delta
        np.add.at(counts, (iy, ix), 1)

    return counts, total_rows, used_events

def plot_pixel_map(counts: np.ndarray, max_delta: int, out_png: Path):
    deltas = np.arange(-max_delta, max_delta + 1)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "figure.dpi": 100,
        }
    )

    # Figura cuadrada; NO usar tight_layout luego
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    extent = [-max_delta - 0.5, max_delta + 0.5, -max_delta - 0.5, max_delta + 0.5]

    im = ax.imshow(
        counts,
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
        extent=extent,
        aspect="equal",  # <- escala 1:1
    )

    # Enforce explícito (por si algún layout intenta “ajustar”)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)  # <- caja del eje cuadrada

    ax.set_xlabel(r"$\Delta x$ (inferior - superior)")
    ax.set_ylabel(r"$\Delta y$ (inferior - superior)")
    ax.set_title("Mapa de píxeles discretos (índices de barra)")

    # Menos ticks para que NO se encimen
    step = 2 if (2 * max_delta + 1) >= 25 else 1
    ticks = np.arange(-max_delta, max_delta + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Colorbar abajo (no estira el ancho)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    cbar.set_label("Número de coincidencias")
    cbar.ax.tick_params(labelsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)  # sin bbox_inches="tight"
    plt.close(fig)


def plot_angular_map(counts: np.ndarray, max_delta: int,
                     pitch_cm: float, distance_cm: float,
                     out_png: Path):

    delta_edges = np.arange(-max_delta - 0.5, max_delta + 1.5, 1.0)
    factor = pitch_cm / distance_cm
    theta_edges_deg = np.degrees(np.arctan(delta_edges * factor))

    theta_x_edges, theta_y_edges = np.meshgrid(theta_edges_deg, theta_edges_deg)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "figure.dpi": 100,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    pcm = ax.pcolormesh(
        theta_x_edges,
        theta_y_edges,
        counts,
        cmap="inferno",
        shading="auto",
    )

    # Enforce 1:1 en grados
    ax.set_xlim(theta_edges_deg[0], theta_edges_deg[-1])
    ax.set_ylim(theta_edges_deg[0], theta_edges_deg[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    ax.set_xlabel(r"$\theta_x$ [deg]")
    ax.set_ylabel(r"$\theta_y$ [deg]")
    ax.set_title("Angular map in ($\\theta_x$, $\\theta_y$)")

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    cbar.set_label("Número de coincidencias")
    cbar.ax.tick_params(labelsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def export_counts_csv(counts: np.ndarray, max_delta: int, out_csv: Path):
    deltas = np.arange(-max_delta, max_delta + 1)
    ny, nx = counts.shape
    assert ny == nx == deltas.size

    dx_grid, dy_grid = np.meshgrid(deltas, deltas)
    flat_counts = counts.ravel()
    flat_dx = dx_grid.ravel()
    flat_dy = dy_grid.ravel()

    mask = flat_counts > 0
    if not np.any(mask):
        return

    df = pd.DataFrame(
        {
            "delta_x": flat_dx[mask],
            "delta_y": flat_dy[mask],
            "counts": flat_counts[mask],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main():
    args = parse_args()
    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    counts, total_rows, used_events = build_pixel_map(
        csv_path,
        chunk_size=args.chunk_size,
        max_delta=args.max_delta,
    )

    print(f"Filas totales leídas: {total_rows}")
    print(f"Eventos usados (1 hit por plano): {used_events}")

    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png_delta = out_dir / "mapa_pixeles_delta_xy.png"
    plot_pixel_map(counts, max_delta=args.max_delta, out_png=out_png_delta)

    out_png_theta = out_dir / "mapa_pixeles_theta_deg.png"
    plot_angular_map(
        counts,
        max_delta=args.max_delta,
        pitch_cm=args.pitch_cm,
        distance_cm=args.distance_cm,
        out_png=out_png_theta,
    )

    out_csv = out_dir / "mapa_pixeles_delta_xy.csv"
    export_counts_csv(counts, max_delta=args.max_delta, out_csv=out_csv)

    print(f"Mapa en Δx,Δy guardado en: {out_png_delta}")
    print(f"Mapa angular (deg) guardado en: {out_png_theta}")
    print(f"Tabla de conteos por píxel guardada en: {out_csv}")


if __name__ == "__main__":
    main()
