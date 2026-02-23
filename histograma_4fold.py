#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)


def parse_channels_from_header(cols):
    """
    Construye un mapa numero->nombre para columnas tipo 'chXX'.
    """
    num2name = {}
    for c in cols:
        m = CH_RE.match(str(c).strip())
        if m:
            n = int(m.group(1))
            num2name[n] = c
    return num2name


def choose_channel_block(num2name, channels_start=1, n_channels=60):
    """
    Devuelve la lista de nombres de columnas chNN para un bloque contiguo
    de tamaño n_channels comenzando en channels_start.
    """
    missing = [
        channels_start + k
        for k in range(n_channels)
        if (channels_start + k) not in num2name
    ]
    if missing:
        raise ValueError(
            f"No encuentro bloque contiguo de {n_channels} canales "
            f"empezando en ch{channels_start:02d}. Faltan, por ejemplo: "
            f"{missing[:10]}"
        )
    return [num2name[channels_start + k] for k in range(n_channels)]


def construir_histograma_coinc4(
    csv_path: Path,
    time_col: str = "time",
    channels_start: int = 1,
    area: float = 0.36,
    chunk_size: int = 100_000,
    n_bars: int = 15,
):
    """
    Lee un CSV (time,chXX...) y construye el histograma del flujo de
    coincidencias 4-fold (un hit en cada uno de los 4 planos) por segundo
    normalizado por el área del panel.
    """

    # --- Detectar columnas y mapa canal->nombre ---
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    if time_col not in header:
        raise ValueError(f"No se encontró la columna temporal '{time_col}'.")

    num2name = parse_channels_from_header(header)
    if not num2name:
        raise ValueError("No se encontraron columnas tipo 'chNN' en el CSV.")

    ch_cols = choose_channel_block(num2name, channels_start=channels_start, n_channels=60)
    if len(ch_cols) != 60:
        raise RuntimeError("Bloque de canales inesperado (no son 60).")

    # Definimos los 4 planos (15 canales cada uno)
    if 4 * n_bars != len(ch_cols):
        raise ValueError(
            f"n_bars={n_bars} inconsistente con {len(ch_cols)} canales "
            f"(se esperan 4*n_bars)."
        )

    g1_cols = ch_cols[0:n_bars]                # X top
    g2_cols = ch_cols[n_bars:2 * n_bars]       # Y top
    g3_cols = ch_cols[2 * n_bars:3 * n_bars]   # X bottom
    g4_cols = ch_cols[3 * n_bars:4 * n_bars]   # Y bottom

    # --- Lectura en chunks y conteo por segundo ---
    usecols = [time_col] + ch_cols
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)

    coinc_per_sec_global = None
    t_start = None
    t_end = None

    if tqdm is not None:
        iterator = tqdm(reader, desc=f"Procesando {csv_path.name}", unit="chunk")
    else:
        iterator = reader

    for chunk in iterator:
        # Convertir time a datetime y filtrar filas válidas
        t = pd.to_datetime(chunk[time_col], errors="coerce")
        mask_t = t.notna()
        if not mask_t.any():
            continue

        chunk = chunk.loc[mask_t].copy()
        t = t.loc[mask_t]

        # Tiempo redondeado al segundo
        t_sec = t.dt.floor("S")
        chunk[time_col] = t_sec

        # Actualizar ventana temporal global
        tmin = t_sec.min()
        tmax = t_sec.max()
        if t_start is None or tmin < t_start:
            t_start = tmin
        if t_end is None or tmax > t_end:
            t_end = tmax

        # Convertir canales a numpy (int8)
        arr = chunk[ch_cols].to_numpy(dtype=np.int8)

        # Índices de columnas en el array
        idx_map = {name: i for i, name in enumerate(ch_cols)}
        g1_idx = [idx_map[c] for c in g1_cols]
        g2_idx = [idx_map[c] for c in g2_cols]
        g3_idx = [idx_map[c] for c in g3_cols]
        g4_idx = [idx_map[c] for c in g4_cols]

        s1 = arr[:, g1_idx]
        s2 = arr[:, g2_idx]
        s3 = arr[:, g3_idx]
        s4 = arr[:, g4_idx]

        c1 = s1.sum(axis=1)
        c2 = s2.sum(axis=1)
        c3 = s3.sum(axis=1)
        c4 = s4.sum(axis=1)

        # Coincidencia 4-fold: exactamente 1 hit por plano
        mask4 = (c1 == 1) & (c2 == 1) & (c3 == 1) & (c4 == 1)
        # si preferieras >=1, cambia por:
        # mask4 = (c1 >= 1) & (c2 >= 1) & (c3 >= 1) & (c4 >= 1)

        if not np.any(mask4):
            continue

        coinc = mask4.astype(int)

        # Serie temporal: coincidencias por segundo en este chunk
        df_tmp = pd.DataFrame(
            {
                time_col: t_sec.to_numpy(),
                "coinc4": coinc,
            }
        )
        grp = df_tmp.groupby(time_col, sort=False)["coinc4"].sum()

        if coinc_per_sec_global is None:
            coinc_per_sec_global = grp
        else:
            coinc_per_sec_global = coinc_per_sec_global.add(grp, fill_value=0.0)

    if coinc_per_sec_global is None or coinc_per_sec_global.empty:
        raise RuntimeError("No se encontraron coincidencias 4-fold en el archivo.")

    # --- Construir histograma de flujo ---
    # Por seguridad, recortamos valores absurdos (no debería haber)
    mask_valid = coinc_per_sec_global.between(0, 3600)
    coinc_valid = coinc_per_sec_global[mask_valid]

    # Flujo 4-fold por segundo y por m^2
    rate4 = (coinc_valid / area).to_numpy(dtype=float)
    N = rate4.size

    mean = float(rate4.mean()) if N > 0 else float("nan")
    std = float(rate4.std(ddof=1)) if N > 1 else float("nan")
    err = std / np.sqrt(N) if N > 0 and np.isfinite(std) else float("nan")

    # Tiempo de exposición total
    exposure_s = (t_end - t_start).total_seconds()
    total_coinc = int(coinc_per_sec_global.sum())

    print("\n=== Estadísticas coincidencias 4-fold ===")
    print(f"Archivo          : {csv_path.name}")
    print(f"Intervalo tiempo : {t_start}  ->  {t_end}")
    print(f"Tiempo efectivo  : {exposure_s:.3f} s")
    print(f"Total coincid.   : {total_coinc}")
    print(f"N segundos usados: {N}")
    print(f"Flujo medio      : {mean:.6f} cnt/s/m^2")
    print(f"Sigma            : {std:.6f} cnt/s/m^2")
    print(f"Error de la media: {err:.6f} cnt/s/m^2")

    # --- Histograma con estética "revista" ---
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "figure.dpi": 100,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    n_bins = 20
    ax.hist(
        rate4,
        bins=n_bins,
        histtype="stepfilled",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_xlabel(r"4-fold flux [s$^{-1}$ m$^{-2}$]")
    ax.set_ylabel("Counts per bin")
    ax.set_title("Histogram of 4-fold flux per second")

    # Línea vertical en la media
    ax.axvline(mean, color="red", linestyle="-", linewidth=1.5, label="Mean")

    # Banda de 1 sigma
    ax.axvspan(
        mean - std,
        mean + std,
        color="red",
        alpha=0.12,
        label=r"$\pm 1\sigma$",
    )

    # Texto con valores
    textstr = (
        rf"$\mu = {mean:.2f}$ s$^{{-1}}$ m$^{{-2}}$" + "\n"
        rf"$\sigma = {std:.2f}$" + "\n"
        rf"$\mu \pm \mathrm{{err}} = {mean:.2f} \pm {err:.2f}$"
    )
    ax.text(
        0.98,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "histograma_flux_4fold.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nHistograma guardado en: {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construye el histograma del flujo de coincidencias 4-fold "
            "(1 hit en cada uno de los 4 planos) a partir de un CSV "
            "time,chXX..., normalizado por el área."
        )
    )
    parser.add_argument("archivo", help="CSV de entrada (time,ch00..ch63)")
    parser.add_argument(
        "--time-col",
        default="time",
        help="Nombre de la columna temporal (defecto: 'time').",
    )
    parser.add_argument(
        "--channels-start",
        type=int,
        default=1,
        help="Canal inicial del bloque contiguo de 60 canales (defecto: 1).",
    )
    parser.add_argument(
        "--area",
        type=float,
        default=0.36,
        help="Área del panel en m^2 (defecto: 0.36).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Filas por chunk para lectura en streaming (defecto: 100000).",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=15,
        help="Número de barras por plano (defecto: 15).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    construir_histograma_coinc4(
        csv_path,
        time_col=args.time_col,
        channels_start=args.channels_start,
        area=args.area,
        chunk_size=args.chunk_size,
        n_bars=args.n_bars,
    )


if __name__ == "__main__":
    main()
