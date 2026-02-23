#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara dos muogramas (fondo y objeto) a nivel de píxeles Δx,Δy "
            "usando canales 1..60 y genera mapas de diferencia de flujo y "
            "atenuación, tanto en índices Δ como en ángulos (deg)."
        )
    )
    parser.add_argument("fondo", help="CSV del fondo (time,ch00..ch63)")
    parser.add_argument("objeto", help="CSV del objetivo (time,ch00..ch63)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Filas por chunk para procesar en streaming (defecto: 200000).",
    )
    parser.add_argument(
        "--max-delta",
        type=int,
        default=14,
        help="Máximo |Δ| en barras (defecto: 14 para 15 barras por plano).",
    )
    parser.add_argument(
        "--pitch-cm",
        type=float,
        default=4.0,
        help="Pitch (ancho de barra) en cm (defecto: 4.0 cm).",
    )
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=70.0,
        help="Distancia entre paneles en cm (defecto: 100.0 cm).",
    )
    parser.add_argument(
        "--area-m2",
        type=float,
        default=0.36,
        help="Área efectiva total de cada panel en m^2 (defecto: 0.36).",
    )
    return parser.parse_args()


def build_pixel_map(csv_path: Path, chunk_size: int, max_delta: int):
    """
    Igual que en angulo.py: construye mapa N(Δx,Δy) a partir del CSV crudo.
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
        iterator = tqdm(
            reader,
            desc=f"Procesando {csv_path.name}",
            unit="chunk",
            dynamic_ncols=True,
        )
    else:
        iterator = reader

    total_rows = 0
    used_events = 0
    t_start = None
    t_end = None

    for chunk in iterator:
        total_rows += len(chunk)

        ts = pd.to_datetime(chunk["time"], errors="coerce")
        if ts.notna().any():
            tmin = ts.min()
            tmax = ts.max()
            if t_start is None or tmin < t_start:
                t_start = tmin
            if t_end is None or tmax > t_end:
                t_end = tmax

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

    if t_start is None or t_end is None:
        raise RuntimeError(
            f"No se pudo determinar la ventana temporal en {csv_path.name}."
        )

    exposure_s = (t_end - t_start).total_seconds()
    return counts, total_rows, used_events, t_start, t_end, exposure_s


def _set_common_rcparams():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "figure.dpi": 100,
        }
    )


def plot_map_delta(matrix: np.ndarray, max_delta: int, out_png: Path,
                   title: str, cbar_label: str,
                   cmap: str = "inferno",
                   vmin=None, vmax=None):
    deltas = np.arange(-max_delta, max_delta + 1)
    _set_common_rcparams()

    fig, ax = plt.subplots(figsize=(7, 6))

    extent = [
        deltas[0] - 0.5,
        deltas[-1] + 0.5,
        deltas[0] - 0.5,
        deltas[-1] + 0.5,
    ]

    im = ax.imshow(
        matrix,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"$\Delta x$ (barra inferior - barra superior)")
    ax.set_ylabel(r"$\Delta y$ (barra inferior - barra superior)")
    ax.set_title(title)

    ax.set_xticks(deltas)
    ax.set_yticks(deltas)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_map_theta(matrix: np.ndarray, max_delta: int,
                   pitch_cm: float, distance_cm: float,
                   out_png: Path,
                   title: str, cbar_label: str,
                   cmap: str = "inferno",
                   vmin=None, vmax=None):
    """
    Representa el mapa en ejes angulares θx,θy (grados).
    """
    delta_edges = np.arange(-max_delta - 0.5, max_delta + 1.5, 1.0)

    factor = pitch_cm / distance_cm
    theta_edges_rad = np.arctan(delta_edges * factor)
    theta_edges_deg = np.degrees(theta_edges_rad)

    theta_x_edges, theta_y_edges = np.meshgrid(theta_edges_deg, theta_edges_deg)

    _set_common_rcparams()
    fig, ax = plt.subplots(figsize=(7, 6))

    pcm = ax.pcolormesh(
        theta_x_edges,
        theta_y_edges,
        matrix,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"$\theta_x$ [deg]")
    ax.set_ylabel(r"$\theta_y$ [deg]")
    ax.set_title(title)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_map_csv(matrix: np.ndarray, max_delta: int, out_csv: Path,
                   value_name: str):
    deltas = np.arange(-max_delta, max_delta + 1)
    ny, nx = matrix.shape
    assert ny == nx == deltas.size

    dx_grid, dy_grid = np.meshgrid(deltas, deltas)
    flat_vals = matrix.ravel()
    flat_dx = dx_grid.ravel()
    flat_dy = dy_grid.ravel()

    df = pd.DataFrame(
        {
            "delta_x": flat_dx,
            "delta_y": flat_dy,
            value_name: flat_vals,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main():
    args = parse_args()
    fondo_path = Path(args.fondo).expanduser().resolve()
    objeto_path = Path(args.objeto).expanduser().resolve()

    if not fondo_path.exists():
        raise FileNotFoundError(fondo_path)
    if not objeto_path.exists():
        raise FileNotFoundError(objeto_path)

    max_delta = args.max_delta

    # --- Fondo ---
    counts_fondo, rows_fondo, events_fondo, t0_f, t1_f, T_f = build_pixel_map(
        fondo_path,
        chunk_size=args.chunk_size,
        max_delta=max_delta,
    )
    if T_f <= 0:
        raise RuntimeError("Tiempo de exposición del fondo no válido.")

    phi_fondo = counts_fondo.astype(float) / (T_f * args.area_m2)

    # --- Objetivo ---
    counts_obj, rows_obj, events_obj, t0_o, t1_o, T_o = build_pixel_map(
        objeto_path,
        chunk_size=args.chunk_size,
        max_delta=max_delta,
    )
    if T_o <= 0:
        raise RuntimeError("Tiempo de exposición del objetivo no válido.")

    phi_obj = counts_obj.astype(float) / (T_o * args.area_m2)

    print("[FONDO]")
    print(f"  Archivo: {fondo_path.name}")
    print(f"  Filas leídas  = {rows_fondo}")
    print(f"  Eventos usados = {events_fondo}")
    print(f"  Ventana: {t0_f} -> {t1_f}")
    print(f"  Tiempo ~ {T_f:.3f} s")
    print("[OBJETO]")
    print(f"  Archivo: {objeto_path.name}")
    print(f"  Filas leídas  = {rows_obj}")
    print(f"  Eventos usados = {events_obj}")
    print(f"  Ventana: {t0_o} -> {t1_o}")
    print(f"  Tiempo ~ {T_o:.3f} s")

    # --- Diferencia de flujo y atenuación ---
    diff_flux = phi_obj - phi_fondo

    with np.errstate(divide="ignore", invalid="ignore"):
        attenuation = diff_flux / phi_fondo
        attenuation[~np.isfinite(attenuation)] = np.nan

    # Directorio de salida
    out_dir = fondo_path.parent / f"graficas_{objeto_path.stem}_vs_{fondo_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diferencia de flujo en Δ
    max_abs = np.nanmax(np.abs(diff_flux)) if np.any(np.isfinite(diff_flux)) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    diff_delta_png = out_dir / "diff_flux_delta_xy.png"
    diff_delta_csv = out_dir / "diff_flux_delta_xy.csv"

    plot_map_delta(
        diff_flux,
        max_delta=max_delta,
        out_png=diff_delta_png,
        title="Δ flux (objeto - fondo) in ($\\Delta x, \\Delta y$)",
        cbar_label=r"Δ flux [s$^{-1}$ m$^{-2}$ per pixel]",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    export_map_csv(
        diff_flux,
        max_delta=max_delta,
        out_csv=diff_delta_csv,
        value_name="diff_flux",
    )

    # Atenuación en Δ
    att_delta_png = out_dir / "attenuation_delta_xy.png"
    att_delta_csv = out_dir / "attenuation_delta_xy.csv"

    plot_map_delta(
        attenuation,
        max_delta=max_delta,
        out_png=att_delta_png,
        title="Attenuation (objeto - fondo) / fondo in ($\\Delta x, \\Delta y$)",
        cbar_label="Attenuation",
        cmap="coolwarm",
        vmin=-0.20,
        vmax=0.20,
    )
    export_map_csv(
        attenuation,
        max_delta=max_delta,
        out_csv=att_delta_csv,
        value_name="attenuation",
    )

    # Diferencia de flujo en ángulo
    diff_theta_png = out_dir / "diff_flux_theta_deg.png"
    plot_map_theta(
        diff_flux,
        max_delta=max_delta,
        pitch_cm=args.pitch_cm,
        distance_cm=args.distance_cm,
        out_png=diff_theta_png,
        title="Δ flux (objeto - fondo) in ($\\theta_x, \\theta_y$)",
        cbar_label=r"Δ flux [s$^{-1}$ m$^{-2}$ per pixel]",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=0.001,
    )

    # Atenuación en ángulo
    att_theta_png = out_dir / "attenuation_theta_deg.png"
    plot_map_theta(
        attenuation,
        max_delta=max_delta,
        pitch_cm=args.pitch_cm,
        distance_cm=args.distance_cm,
        out_png=att_theta_png,
        title="Attenuation (objeto - fondo) / fondo in ($\\theta_x, \\theta_y$)",
        cbar_label="Attenuation",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    print(f"\nMapas de Δ flux guardados en: {diff_delta_png} y {diff_theta_png}")
    print(f"Mapas de atenuación guardados en: {att_delta_png} y {att_theta_png}")
    print("CSV de Δ flux: ", diff_delta_csv)
    print("CSV de atenuación: ", att_delta_csv)
    print("\nOJO: esto está normalizado por tiempo y área del panel, "
          "pero todavía no por aceptación en sólido angular de cada píxel.")


if __name__ == "__main__":
    main()
