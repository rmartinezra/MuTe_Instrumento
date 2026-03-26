#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Construye mapa N(Δx,Δy) y mapa angular (θx,θy) "
            "desde coincidencias 4 planos (X1,Y1,X2,Y2) en CSV time,ch00..ch63.\n\n"
            "Mapeo (nuevo):\n"
            "  Panel 1 abajo:  X1=ch01..ch15, Y1=ch16..ch30\n"
            "  Panel 2 arriba: X2=ch32..ch46, Y2=ch47..ch61  (ch31 sin uso)\n"
            "Definición: Δx=iX2−iX1, Δy=iY2−iY1"
        )
    )
    p.add_argument("archivo", help="CSV de entrada (time,ch00..ch63)")
    p.add_argument("--chunk-size", type=int, default=200_000,
                   help="Filas por chunk (defecto: 200000)")
    p.add_argument("--max-delta", type=int, default=14,
                   help="Máximo |Δ| en barras (defecto: 14 para 15 barras)")
    p.add_argument("--pitch-cm", type=float, default=4.0,
                   help="Pitch (ancho de barra) en cm (defecto: 4.0)")
    p.add_argument("--distance-cm", type=float, default=30.0,
                   help="Separación entre paneles en cm (defecto: 30.0)")
    p.add_argument("--strict-single-hit", action="store_true",
                   help="Si se activa: exige EXACTAMENTE 1 hit por plano (si no, usa centroid con multihit).")
    p.add_argument("--flip-y2", action="store_true",
                   help="Aplica iY2 -> (14 - iY2). Útil solo si alguna corrida tiene Y2 invertido.")
    return p.parse_args()


# -----------------------------
# Core
# -----------------------------
def _centroid_0_14(plane_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    plane_arr: shape (N, 15) con 0/1 (o conteos).
    Retorna:
      - centroid (float) en [0,14] para filas con sum>0, NaN si sum==0
      - multiplicidad (int) = sum por fila
    """
    mult = plane_arr.sum(axis=1).astype(np.int32)
    idx = np.arange(plane_arr.shape[1], dtype=np.float32)  # 0..14
    num = plane_arr @ idx
    cen = np.full(plane_arr.shape[0], np.nan, dtype=np.float32)
    m = mult > 0
    cen[m] = num[m] / mult[m]
    return cen, mult


def build_pixel_map(csv_path: Path, chunk_size: int, max_delta: int,
                    strict_single_hit: bool, flip_y2: bool):
    """
    Acumula counts[Δy,Δx] para Δx,Δy enteros en [-max_delta, +max_delta].

    Mapeo (nuevo):
      X1 = ch01..ch15   (panel 1 abajo)
      Y1 = ch16..ch30
      X2 = ch32..ch46   (panel 2 arriba)
      Y2 = ch47..ch61
      ch00, ch31, ch62..ch63 ignorados
    """
    side = 2 * max_delta + 1
    counts = np.zeros((side, side), dtype=np.int64)

    # columnas ch00..ch63
    ch_cols = [f"ch{i:02d}" for i in range(64)]
    usecols = ["time"] + ch_cols

    # indices en la matriz arr (porque arr incluye ch00 en la col 0)
    gX1 = np.arange(1, 16)     # ch01..ch15
    gY1 = np.arange(16, 31)    # ch16..ch30
    gX2 = np.arange(32, 47)    # ch32..ch46
    gY2 = np.arange(47, 62)    # ch47..ch61

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)

    iterator = tqdm(reader, desc="Procesando eventos", unit="chunk", dynamic_ncols=True) if tqdm else reader

    total_rows = 0
    used_events = 0

    for chunk in iterator:
        total_rows += len(chunk)

        # (N,64) con ch00..ch63
        arr = chunk[ch_cols].to_numpy(dtype=np.int8)

        sX1 = arr[:, gX1]
        sY1 = arr[:, gY1]
        sX2 = arr[:, gX2]
        sY2 = arr[:, gY2]

        iX1, mX1 = _centroid_0_14(sX1)
        iY1, mY1 = _centroid_0_14(sY1)
        iX2, mX2 = _centroid_0_14(sX2)
        iY2, mY2 = _centroid_0_14(sY2)

        # requiere al menos 1 hit por plano
        mask = (mX1 > 0) & (mY1 > 0) & (mX2 > 0) & (mY2 > 0)

        # si se pide estricto: exactamente 1 hit por plano
        if strict_single_hit:
            mask &= (mX1 == 1) & (mY1 == 1) & (mX2 == 1) & (mY2 == 1)

        if not np.any(mask):
            continue

        x1 = iX1[mask]
        y1 = iY1[mask]
        x2 = iX2[mask]
        y2 = iY2[mask]

        if flip_y2:
            y2 = (14.0 - y2)

        dx = x2 - x1
        dy = y2 - y1

        # Para el mapa de píxeles discretos, binning entero
        dx_i = np.rint(dx).astype(np.int16)
        dy_i = np.rint(dy).astype(np.int16)

        valid = (
            (dx_i >= -max_delta) & (dx_i <= max_delta) &
            (dy_i >= -max_delta) & (dy_i <= max_delta)
        )
        if not np.any(valid):
            continue

        dx_i = dx_i[valid]
        dy_i = dy_i[valid]

        used_events += dx_i.size

        ix = dx_i + max_delta
        iy = dy_i + max_delta
        np.add.at(counts, (iy, ix), 1)

    return counts, total_rows, used_events


# -----------------------------
# Plots + export
# -----------------------------
def plot_pixel_map(counts: np.ndarray, max_delta: int, out_png: Path):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14, "figure.dpi": 100})
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    extent = [-max_delta - 0.5, max_delta + 0.5, -max_delta - 0.5, max_delta + 0.5]
    im = ax.imshow(counts, origin="lower", interpolation="nearest", extent=extent, aspect="equal")

    ax.set_xlabel(r"$\Delta x = i_{X2}-i_{X1}$")
    ax.set_ylabel(r"$\Delta y = i_{Y2}-i_{Y1}$")
    ax.set_title("Mapa de píxeles discretos N(Δx,Δy)")

    step = 2 if (2 * max_delta + 1) >= 25 else 1
    ticks = np.arange(-max_delta, max_delta + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    cbar.set_label("Número de coincidencias")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_angular_map(counts: np.ndarray, max_delta: int, pitch_cm: float, distance_cm: float, out_png: Path):
    delta_edges = np.arange(-max_delta - 0.5, max_delta + 1.5, 1.0)  # bordes en Δ (paso 1)
    factor = pitch_cm / distance_cm
    theta_edges_deg = np.degrees(np.arctan(delta_edges * factor))

    theta_x_edges, theta_y_edges = np.meshgrid(theta_edges_deg, theta_edges_deg)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14, "figure.dpi": 100})
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    counts_plot = np.ma.masked_less_equal(counts, 0)
    pcm = ax.pcolormesh(
        theta_x_edges, theta_y_edges, counts_plot,
        shading="auto", norm=LogNorm()
    )

    ax.set_xlabel(r"$\theta_x$ [deg]")
    ax.set_ylabel(r"$\theta_y$ [deg]")
    ax.set_title("Mapa angular N($\\theta_x$, $\\theta_y$)")

    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    cbar.set_label("Número de coincidencias")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def export_counts_csv(counts: np.ndarray, max_delta: int, out_csv: Path):
    deltas = np.arange(-max_delta, max_delta + 1)
    dx_grid, dy_grid = np.meshgrid(deltas, deltas)

    flat_counts = counts.ravel()
    flat_dx = dx_grid.ravel()
    flat_dy = dy_grid.ravel()

    m = flat_counts > 0
    if not np.any(m):
        return

    df = pd.DataFrame({"delta_x": flat_dx[m], "delta_y": flat_dy[m], "counts": flat_counts[m]})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    counts, total_rows, used_events = build_pixel_map(
        csv_path,
        chunk_size=args.chunk_size,
        max_delta=args.max_delta,
        strict_single_hit=args.strict_single_hit,
        flip_y2=args.flip_y2,
    )

    print(f"Filas leídas: {total_rows}")
    print(f"Eventos usados (4 planos, criterio seleccionado): {used_events}")
    if args.strict_single_hit:
        print("Modo: strict-single-hit (exactamente 1 hit por plano).")
    else:
        print("Modo: centroid (permite multihit; centroid por plano).")
    print(f"Flip Y2: {'sí' if args.flip_y2 else 'no'}")

    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png_delta = out_dir / "mapa_pixeles_delta_xy.png"
    plot_pixel_map(counts, max_delta=args.max_delta, out_png=out_png_delta)

    out_png_theta = out_dir / "mapa_pixeles_theta_deg.png"
    plot_angular_map(counts, max_delta=args.max_delta,
                     pitch_cm=args.pitch_cm, distance_cm=args.distance_cm,
                     out_png=out_png_theta)

    out_csv = out_dir / "mapa_pixeles_delta_xy.csv"
    export_counts_csv(counts, max_delta=args.max_delta, out_csv=out_csv)

    print(f"PNG ΔxΔy: {out_png_delta}")
    print(f"PNG θxθy: {out_png_theta}")
    print(f"CSV conteos: {out_csv}")


if __name__ == "__main__":
    main()
