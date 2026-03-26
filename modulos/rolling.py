#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)


def parse_channels_from_header(cols):
    """Construye un mapa numero->nombre para columnas tipo chNN."""
    num2name = {}
    for c in cols:
        m = CH_RE.match(str(c).strip())
        if m:
            num2name[int(m.group(1))] = c
    return num2name


def choose_channel_block(num2name, channels_start=1, n_channels=60):
    """Devuelve los nombres de columnas de un bloque contiguo chNN."""
    missing = [
        channels_start + k
        for k in range(n_channels)
        if (channels_start + k) not in num2name
    ]
    if missing:
        raise ValueError(
            f"No encuentro bloque contiguo de {n_channels} canales "
            f"empezando en ch{channels_start:02d}. Faltan, por ejemplo: {missing[:10]}"
        )
    return [num2name[channels_start + k] for k in range(n_channels)]


def detectar_coincidencias_4fold(chunk, ch_cols, n_bars):
    """
    Devuelve un array 0/1 por fila indicando coincidencia 4-fold:
    exactamente un hit en cada uno de los 4 planos.
    """
    if len(ch_cols) != 4 * n_bars:
        raise ValueError(
            f"n_bars={n_bars} inconsistente con {len(ch_cols)} canales. "
            f"Se esperan 4*n_bars canales."
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

    mask4 = (c1 == 1) & (c2 == 1) & (c3 == 1) & (c4 == 1)
    return mask4.astype(np.int8)


def construir_series_temporales(
    csv_path: Path,
    time_col: str = "time",
    channels_start: int = 1,
    area: float = 0.36,
    chunk_size: int = 100_000,
    n_bars: int = 15,
):
    """
    Construye una serie de conteos 4-fold por segundo y una serie de conteos
    por minuto derivada de la serie por segundo.

    La definición correcta para promedios/histogramas por segundo rellena con 0
    los segundos sin eventos. Eso evita sesgar el promedio al alza.
    """
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    if time_col not in header:
        raise ValueError(f"No se encontró la columna temporal '{time_col}'.")

    num2name = parse_channels_from_header(header)
    if not num2name:
        raise ValueError("No se encontraron columnas tipo 'chNN' en el CSV.")

    ch_cols = choose_channel_block(num2name, channels_start=channels_start, n_channels=60)
    usecols = [time_col] + ch_cols
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)

    coinc_per_sec_global = None
    t_start = None
    t_end = None

    iterator = tqdm(reader, desc=f"Procesando {csv_path.name}", unit="chunk") if tqdm is not None else reader

    for chunk in iterator:
        t = pd.to_datetime(chunk[time_col], errors="coerce")
        mask_t = t.notna()
        if not mask_t.any():
            continue

        chunk = chunk.loc[mask_t].copy()
        t = t.loc[mask_t]
        t_sec = t.dt.floor("s")

        tmin = t_sec.min()
        tmax = t_sec.max()
        if t_start is None or tmin < t_start:
            t_start = tmin
        if t_end is None or tmax > t_end:
            t_end = tmax

        coinc = detectar_coincidencias_4fold(chunk, ch_cols=ch_cols, n_bars=n_bars)
        if not np.any(coinc):
            continue

        df_tmp = pd.DataFrame({time_col: t_sec.to_numpy(), "coinc4": coinc})
        grp = df_tmp.groupby(time_col, sort=False)["coinc4"].sum()

        if coinc_per_sec_global is None:
            coinc_per_sec_global = grp
        else:
            coinc_per_sec_global = coinc_per_sec_global.add(grp, fill_value=0)

    if coinc_per_sec_global is None:
        raise RuntimeError("No se encontraron coincidencias 4-fold en el archivo.")

    coinc_per_sec_global = coinc_per_sec_global.sort_index()

    # Serie completa por segundo, incluyendo segundos sin eventos.
    full_second_index = pd.date_range(start=t_start.floor("s"), end=t_end.floor("s"), freq="s")
    counts_per_sec = coinc_per_sec_global.reindex(full_second_index, fill_value=0).astype(int)
    counts_per_sec.name = "counts_4fold_per_sec"

    # Serie por minuto. Idealmente usamos solo minutos completos para evitar que
    # el primero o último minuto parcial distorsionen el histograma.
    first_ts = counts_per_sec.index[0]
    last_ts = counts_per_sec.index[-1]
    first_full_min = first_ts if first_ts.second == 0 else first_ts.ceil("min")
    last_full_min = (last_ts - pd.Timedelta(seconds=59)).floor("min")

    minute_all = counts_per_sec.resample("min").sum()
    if first_full_min <= last_full_min:
        counts_per_min = minute_all.loc[first_full_min:last_full_min].copy()
        minute_mode = "solo minutos completos"
    else:
        counts_per_min = minute_all.copy()
        minute_mode = "minutos parciales incluidos (duración total < 60 s)"
    counts_per_min.name = "counts_4fold_per_min"

    stats = {
        "archivo": csv_path.name,
        "t_start": t_start,
        "t_end": t_end,
        "n_seconds": int(len(counts_per_sec)),
        "n_minutes": int(len(counts_per_min)),
        "minute_mode": minute_mode,
        "total_coinc": int(counts_per_sec.sum()),
        "mean_cps": float(counts_per_sec.mean()),
        "std_cps": float(counts_per_sec.std(ddof=1)) if len(counts_per_sec) > 1 else float("nan"),
        "mean_flux": float(counts_per_sec.mean() / area),
        "std_flux": float(counts_per_sec.std(ddof=1) / area) if len(counts_per_sec) > 1 else float("nan"),
        "mean_cpm": float(counts_per_min.mean()) if len(counts_per_min) > 0 else float("nan"),
        "std_cpm": float(counts_per_min.std(ddof=1)) if len(counts_per_min) > 1 else float("nan"),
        "area": float(area),
    }

    return counts_per_sec, counts_per_min, stats


def rolling_mean_and_error(series, window_points):
    roll_mean = series.rolling(window=window_points, min_periods=1).mean()
    roll_std = series.rolling(window=window_points, min_periods=4).std(ddof=1)
    roll_n = series.rolling(window=window_points, min_periods=1).count()
    roll_err = roll_std / np.sqrt(roll_n)
    return roll_mean, roll_err


def plot_time_series_with_rolling(series, ylabel, title, out_png, window_points, raw_alpha=0.45):
    roll_mean, roll_err = rolling_mean_and_error(series, window_points)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(series.index, series.to_numpy(), linewidth=0.7, alpha=raw_alpha, label="Raw series")
    ax.plot(series.index, roll_mean.to_numpy(), linewidth=1.8, label=f"Rolling mean ({window_points} bins)")

    y_lower = (roll_mean - roll_err).to_numpy()
    y_upper = (roll_mean + roll_err).to_numpy()
    ax.fill_between(series.index, y_lower, y_upper, alpha=0.25, label="Rolling mean ± error")

    ax.axhline(series.mean(), linestyle="--", linewidth=1.2, label="Global mean")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(series, xlabel, title, out_png, bins=9):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    values = series.to_numpy(dtype=float)
    mean = float(np.mean(values)) if len(values) else float("nan")
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    err = std / np.sqrt(len(values)) if len(values) > 1 else float("nan")

    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.8, edgecolor="black", linewidth=1.0)
    ax.axvline(mean, color="red", linestyle="-", linewidth=1.5, label="Mean")

    if np.isfinite(std):
        ax.axvspan(mean - std, mean + std, color="red", alpha=0.12, label=r"$\pm 1\sigma$")

    textstr = (
        rf"$\mu = {mean:.2f}$" + "\n" +
        rf"$\sigma = {std:.2f}$" + "\n" +
        rf"$\mu \pm \mathrm{{err}} = {mean:.2f} \pm {err:.2f}$"
    )
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=11,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts per bin")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def guardar_series_csv(counts_per_sec, counts_per_min, out_dir, area):
    df_sec = pd.DataFrame({
        "time": counts_per_sec.index,
        "counts_4fold_per_sec": counts_per_sec.to_numpy(),
        "flux_4fold_per_sec_m2": counts_per_sec.to_numpy(dtype=float) / area,
    })
    df_min = pd.DataFrame({
        "time": counts_per_min.index,
        "counts_4fold_per_min": counts_per_min.to_numpy(),
        "flux_4fold_per_min_m2": counts_per_min.to_numpy(dtype=float) / area,
    })
    sec_csv = out_dir / "serie_4fold_por_segundo.csv"
    min_csv = out_dir / "serie_4fold_por_minuto.csv"
    df_sec.to_csv(sec_csv, index=False)
    df_min.to_csv(min_csv, index=False)
    return sec_csv, min_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unifica el cálculo 4-fold en un solo script, usando la definición correcta "
            "de conteos por segundo: incluir segundos sin eventos. Además genera gráficas "
            "por segundo y por minuto."
        )
    )
    parser.add_argument("archivo", help="CSV de entrada (time,chXX...)")
    parser.add_argument("--time-col", default="time", help="Nombre de la columna temporal.")
    parser.add_argument("--channels-start", type=int, default=1, help="Canal inicial del bloque contiguo de 60 canales.")
    parser.add_argument("--area", type=float, default=0.36, help="Área del panel en m^2.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Filas por chunk.")
    parser.add_argument("--n-bars", type=int, default=15, help="Número de barras por plano.")
    parser.add_argument("--rolling-sec-window", type=int, default=10, help="Ventana rolling para la serie por segundo.")
    parser.add_argument("--rolling-min-window", type=int, default=10, help="Ventana rolling para la serie por minuto.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "figure.dpi": 100,
        }
    )

    counts_per_sec, counts_per_min, stats = construir_series_temporales(
        csv_path=csv_path,
        time_col=args.time_col,
        channels_start=args.channels_start,
        area=args.area,
        chunk_size=args.chunk_size,
        n_bars=args.n_bars,
    )

    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_time_series_with_rolling(
        counts_per_sec,
        ylabel="4-fold counts per second",
        title="4-fold counts per second vs time",
        out_png=out_dir / "conteos_4fold_por_segundo_tiempo.png",
        window_points=args.rolling_sec_window,
    )
    plot_histogram(
        counts_per_sec,
        xlabel="4-fold counts per second",
        title="Histogram of 4-fold counts per second",
        out_png=out_dir / "histograma_conteos_4fold_por_segundo.png",
        bins=14,
    )
    plot_time_series_with_rolling(
        counts_per_min,
        ylabel="4-fold counts per minute",
        title="4-fold counts per minute vs time",
        out_png=out_dir / "conteos_4fold_por_minuto_tiempo.png",
        window_points=args.rolling_min_window,
        raw_alpha=0.55,
    )
    plot_histogram(
        counts_per_min,
        xlabel="4-fold counts per minute",
        title="Histogram of 4-fold counts per minute",
        out_png=out_dir / "histograma_conteos_4fold_por_minuto.png",
        bins=14,
    )

    sec_csv, min_csv = guardar_series_csv(counts_per_sec, counts_per_min, out_dir, args.area)

    print("\n=== Resumen 4-fold unificado ===")
    print(f"Archivo                 : {stats['archivo']}")
    print(f"Intervalo               : {stats['t_start']} -> {stats['t_end']}")
    print(f"Total coincidencias     : {stats['total_coinc']}")
    print(f"N segundos              : {stats['n_seconds']}")
    print(f"N minutos               : {stats['n_minutes']} ({stats['minute_mode']})")
    print(f"Media conteos/s         : {stats['mean_cps']:.6f}")
    print(f"Sigma conteos/s         : {stats['std_cps']:.6f}")
    print(f"Media flujo/s/m^2       : {stats['mean_flux']:.6f}")
    print(f"Sigma flujo/s/m^2       : {stats['std_flux']:.6f}")
    print(f"Media conteos/min       : {stats['mean_cpm']:.6f}")
    print(f"Sigma conteos/min       : {stats['std_cpm']:.6f}")
    print(f"CSV por segundo         : {sec_csv}")
    print(f"CSV por minuto          : {min_csv}")
    print(f"Directorio de salida    : {out_dir}")


if __name__ == "__main__":
    main()
