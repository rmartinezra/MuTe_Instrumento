#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Construye mapa discreto N(Δx,Δy) y mapa angular (θx,θy_global) "
            "desde coincidencias 4 planos (X1,Y1,X2,Y2) en CSV time,ch00..ch63.\n\n"
            "Mapeo:\n"
            "  Panel 1: X1=ch01..ch15, Y1=ch16..ch30\n"
            "  Panel 2: X2=ch32..ch46, Y2=ch47..ch61 (ch31 sin uso)\n"
            "Definición: Δx=iX2−iX1, Δy=iY2−iY1"
        )
    )
    p.add_argument("archivo", help="CSV de entrada (time,ch00..ch63)")
    p.add_argument("--chunk-size", type=int, default=200_000,
                   help="Filas por chunk (defecto: 200000)")
    p.add_argument("--max-delta", type=int, default=14,
                   help="Máximo |Δ| en barras (defecto: 14)")
    p.add_argument("--pitch-cm", type=float, default=4.0,
                   help="Pitch (ancho de barra) en cm (defecto: 4.0)")
    p.add_argument("--distance-cm", type=float, default=200.0,
                   help="Separación entre paneles en cm (defecto: 200.0)")
    p.add_argument("--strict-single-hit", action="store_true",
                   help="Exige exactamente 1 hit por plano; si no, usa centroid.")
    p.add_argument("--invert-y", action="store_true", default=True,
                   help=("Invierte la coordenada física de Y en ambos paneles: "
                         "barra menor = posición más alta. Por defecto: activado."))
    p.add_argument("--no-invert-y", dest="invert_y", action="store_false",
                   help="Desactiva la inversión física de Y en ambos paneles.")
    p.add_argument("--tilt-y-deg", type=float, default=12.0,
                   help="Corrimiento global a aplicar en θy [deg] (defecto: 10.0)")
    p.add_argument("--low-count-threshold", type=int, default=0,
                   help="Bins con cuentas menores a este umbral se dejan en blanco (defecto: 5)")
    return p.parse_args()


def _centroid_0_14(plane_arr: np.ndarray):
    mult = plane_arr.sum(axis=1).astype(np.int32)
    idx = np.arange(plane_arr.shape[1], dtype=np.float32)
    num = plane_arr @ idx
    cen = np.full(plane_arr.shape[0], np.nan, dtype=np.float32)
    m = mult > 0
    cen[m] = num[m] / mult[m]
    return cen, mult


def build_delta_map(csv_path: Path, chunk_size: int, max_delta: int,
                    strict_single_hit: bool, invert_y: bool):
    side = 2 * max_delta + 1
    counts_delta = np.zeros((side, side), dtype=np.int64)

    ch_cols = [f"ch{i:02d}" for i in range(64)]
    usecols = ["time"] + ch_cols

    gX1 = np.arange(1, 16)
    gY1 = np.arange(16, 31)
    gX2 = np.arange(32, 47)
    gY2 = np.arange(47, 62)

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)
    iterator = tqdm(reader, desc="Procesando eventos", unit="chunk", dynamic_ncols=True) if tqdm else reader

    total_rows = 0
    used_events = 0

    for chunk in iterator:
        total_rows += len(chunk)
        arr = chunk[ch_cols].to_numpy(dtype=np.int8)

        sX1 = arr[:, gX1]
        sY1 = arr[:, gY1]
        sX2 = arr[:, gX2]
        sY2 = arr[:, gY2]

        iX1, mX1 = _centroid_0_14(sX1)
        iY1, mY1 = _centroid_0_14(sY1)
        iX2, mX2 = _centroid_0_14(sX2)
        iY2, mY2 = _centroid_0_14(sY2)

        mask = (mX1 > 0) & (mY1 > 0) & (mX2 > 0) & (mY2 > 0)
        if strict_single_hit:
            mask &= (mX1 == 1) & (mY1 == 1) & (mX2 == 1) & (mY2 == 1)

        if not np.any(mask):
            continue

        x1 = iX1[mask]
        y1 = iY1[mask]
        x2 = iX2[mask]
        y2 = iY2[mask]

        if invert_y:
            y1 = 14.0 - y1
            y2 = 14.0 - y2

        x1_i = np.rint(x1).astype(np.int16)
        y1_i = np.rint(y1).astype(np.int16)
        x2_i = np.rint(x2).astype(np.int16)
        y2_i = np.rint(y2).astype(np.int16)

        valid_bar = (
            (x1_i >= 0) & (x1_i <= 14) &
            (x2_i >= 0) & (x2_i <= 14) &
            (y1_i >= 0) & (y1_i <= 14) &
            (y2_i >= 0) & (y2_i <= 14)
        )
        if not np.any(valid_bar):
            continue

        x1_i = x1_i[valid_bar]
        y1_i = y1_i[valid_bar]
        x2_i = x2_i[valid_bar]
        y2_i = y2_i[valid_bar]

        dx_i = x2_i - x1_i
        dy_i = y2_i - y1_i

        valid_delta = (
            (dx_i >= -max_delta) & (dx_i <= max_delta) &
            (dy_i >= -max_delta) & (dy_i <= max_delta)
        )
        if not np.any(valid_delta):
            continue

        dx_i = dx_i[valid_delta]
        dy_i = dy_i[valid_delta]

        used_events += dx_i.size
        ix = dx_i + max_delta
        iy = dy_i + max_delta
        np.add.at(counts_delta, (iy, ix), 1)

    return counts_delta, total_rows, used_events


def _prepare_masked_counts(counts: np.ndarray, low_count_threshold: int):
    threshold = max(1, int(low_count_threshold))
    counts_plot = np.ma.masked_less(counts, threshold)

    cmap = cm.get_cmap("viridis").copy()
    cmap.set_bad("white")
    cmap.set_under("white")

    valid_counts = counts[counts >= threshold]
    if valid_counts.size:
        vmin = threshold
        vmax = int(valid_counts.max())
        if vmax <= vmin:
            vmax = vmin + 1
        norm = LogNorm(vmin=vmin, vmax=vmax)
        has_valid = True
    else:
        norm = Normalize(vmin=threshold, vmax=threshold + 1)
        has_valid = False

    return counts_plot, cmap, norm, threshold, has_valid


def _add_colorbar(fig, ax, cmap, norm, threshold: int, has_valid: bool):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    if has_valid:
        cbar.set_label(f"Número de coincidencias (bins con N < {threshold} se dejan en blanco)")
    else:
        cbar.set_label(f"No hay bins con N ≥ {threshold}")
    return cbar


def plot_pixel_map(counts: np.ndarray, max_delta: int, low_count_threshold: int, out_png: Path):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14, "figure.dpi": 100})
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    counts_plot, cmap, norm, threshold, has_valid = _prepare_masked_counts(counts, low_count_threshold)
    extent = [-max_delta - 0.5, max_delta + 0.5, -max_delta - 0.5, max_delta + 0.5]
    im = ax.imshow(
        counts_plot,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        norm=norm
    )

    ax.set_xlabel(r"$\Delta x$ [barras]")
    ax.set_ylabel(r"$\Delta y$ [barras]")
    ax.set_title("Mapa de píxeles discretos N(Δx,Δy)")

    step = 2 if (2 * max_delta + 1) >= 25 else 1
    ticks = np.arange(-max_delta, max_delta + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    _add_colorbar(fig, ax, cmap, norm, threshold, has_valid)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_angular_map(counts: np.ndarray, max_delta: int, pitch_cm: float,
                     distance_cm: float, tilt_y_deg: float,
                     low_count_threshold: int, out_png: Path):
    delta_edges = np.arange(-max_delta - 0.5, max_delta + 1.5, 1.0)
    factor = pitch_cm / distance_cm
    theta_edges_deg = np.degrees(np.arctan(delta_edges * factor))

    theta_x_edges = theta_edges_deg
    theta_y_edges_global = theta_edges_deg + tilt_y_deg
    theta_x_edges_grid, theta_y_edges_grid = np.meshgrid(theta_x_edges, theta_y_edges_global)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14, "figure.dpi": 100})
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    counts_plot, cmap, norm, threshold, has_valid = _prepare_masked_counts(counts, low_count_threshold)
    pcm = ax.pcolormesh(
        theta_x_edges_grid,
        theta_y_edges_grid,
        counts_plot,
        shading="auto",
        cmap=cmap,
        norm=norm
    )

    ax.set_xlabel(r"$\theta_x$ [deg]")
    ax.set_ylabel(r"$\theta_{y,\mathrm{global}}$ [deg]")
    ax.set_title(f"Mapa angular N($\\theta_x$, $\\theta_y^{{global}}$), tilt Y = {tilt_y_deg:g}°")
    ax.set_aspect("equal", adjustable="box")

    _add_colorbar(fig, ax, cmap, norm, threshold, has_valid)

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


def main():
    args = parse_args()
    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    counts_delta, total_rows, used_events = build_delta_map(
        csv_path,
        chunk_size=args.chunk_size,
        max_delta=args.max_delta,
        strict_single_hit=args.strict_single_hit,
        invert_y=args.invert_y,
    )

    print(f"Filas leídas: {total_rows}")
    print(f"Eventos usados (4 planos, criterio seleccionado): {used_events}")
    print("Modo: strict-single-hit (exactamente 1 hit por plano)." if args.strict_single_hit
          else "Modo: centroid (permite multihit; centroid por plano).")
    print(f"Inversión física de Y: {'sí' if args.invert_y else 'no'}")
    print(f"Tilt global en Y: {args.tilt_y_deg} deg")
    print(f"Umbral visual low-count: N < {args.low_count_threshold} queda en blanco")

    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png_delta = out_dir / "mapa_pixeles_delta_xy.png"
    plot_pixel_map(counts_delta, max_delta=args.max_delta,
                   low_count_threshold=args.low_count_threshold,
                   out_png=out_png_delta)

    out_png_theta = out_dir / "mapa_pixeles_theta_deg_global.png"
    plot_angular_map(
        counts_delta,
        max_delta=args.max_delta,
        pitch_cm=args.pitch_cm,
        distance_cm=args.distance_cm,
        tilt_y_deg=args.tilt_y_deg,
        low_count_threshold=args.low_count_threshold,
        out_png=out_png_theta,
    )

    out_csv = out_dir / "mapa_pixeles_delta_xy.csv"
    export_counts_csv(counts_delta, max_delta=args.max_delta, out_csv=out_csv)

    print(f"PNG ΔxΔy: {out_png_delta}")
    print(f"PNG θxθy global: {out_png_theta}")
    print(f"CSV conteos: {out_csv}")


if __name__ == "__main__":
    main()
