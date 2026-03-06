#!/usr/bin/env python3
"""Analisis unificado (tasa + coincidencias)

Este script integra:
  1) Analisis de tasa/estabilidad por canal (antes: analisischV1.py)
  2) Analisis de coincidencias por activacion (antes: tasaV1.py)

Todas las salidas se guardan en un unico directorio dentro del folder del
CSV de entrada, con el mismo criterio de tasaV1:

  <carpeta_del_csv>/graficas_<nombre_del_csv_sin_extension>/

Notas importantes:
  - Para el analisis de tasa, por defecto SOLO se usan los canales fisicos:
    Panel 1: ch1..ch30
    Panel 2: ch32..ch61
    (se excluye ch31)
  - Para coincidencias se usa la misma geometria fisica anterior:
    P1 = (1..15)x(16..30), P2 = (32..46)x(47..61)
  - Los heatmaps individuales de Panel 1 y Panel 2 se guardan con la misma
    escala de color dentro de cada metrica, para que la comparacion visual
    sea directa.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Utilidades de canales
# -----------------------------

_CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)


def _extract_channels(header_cols: list[str]) -> dict[int, str]:
    """Mapea numero->nombre real para columnas chXX, conservando el primero si hay duplicados."""
    num2name: dict[int, str] = {}
    for c in header_cols:
        m = _CH_RE.match(str(c).strip())
        if m:
            n = int(m.group(1))
            num2name.setdefault(n, c)
    return num2name


def _select_contiguous_channels(num2name: dict[int, str], start: int, n: int) -> list[str]:
    """Selecciona un rango contiguo [start, start+n-1]. Lanza error con lista de faltantes si no existe."""
    want = [start + k for k in range(n)]
    missing = [x for x in want if x not in num2name]
    if missing:
        raise ValueError(
            "No pude construir un rango contiguo de canales. "
            f"Esperaba ch{start}..ch{start+n-1}. "
            f"Faltantes (primeros 20): {missing[:20]}"
        )
    return [num2name[x] for x in want]


def _select_explicit_channel_numbers(num2name: dict[int, str], numbers: list[int]) -> list[str]:
    """Selecciona una lista explicita de numeros de canal, preservando el orden dado."""
    missing = [n for n in numbers if n not in num2name]
    if missing:
        raise ValueError(
            "No pude construir la lista explicita de canales. "
            f"Faltantes (primeros 20): {missing[:20]}"
        )
    return [num2name[n] for n in numbers]


def _default_physical_channel_numbers() -> list[int]:
    """Canales fisicos usados por defecto: P1=1..30, P2=32..61 (excluye 31)."""
    return list(range(1, 31)) + list(range(32, 62))


def _select_channel_list_for_coincidences(num2name: dict[int, str], channels_start: int) -> list[str]:
    """Replica la logica de tasaV1: intenta 60 (preferido) o 64 si existe completo."""
    nums = sorted(num2name.keys())

    if (channels_start + 60 - 1) in nums and all((channels_start + k) in num2name for k in range(60)):
        n_ch = 60
    elif (channels_start + 64 - 1) in nums and all((channels_start + k) in num2name for k in range(64)):
        n_ch = 64
    else:
        # fallback (si está EXACTO)
        if len(nums) == 60 and all((channels_start + k) in num2name for k in range(60)):
            n_ch = 60
        elif len(nums) == 64 and all((channels_start + k) in num2name for k in range(64)):
            n_ch = 64
        else:
            want60 = [channels_start + k for k in range(60)]
            missing = [n for n in want60 if n not in num2name]
            raise ValueError(
                "No pude construir un rango contiguo de canales para coincidencias.\n"
                f"channels_start={channels_start}. Ejemplo esperado: ch{channels_start}..ch{channels_start+59}.\n"
                f"Faltantes (primeros 20): {missing[:20]}"
            )

    return [num2name[channels_start + k] for k in range(n_ch)]


# -----------------------------
# Parte A: Analisis de tasa/estabilidad (analisischV1)
# -----------------------------

def _acumular_cps_y_multiplicidad(
    ruta_csv: Path,
    canales: list[str],
    chunksize: int = 200_000,
):
    """Streaming CSV: acumula conteos por segundo y multiplicidad por evento (sin cargar todo)."""
    n_canales = len(canales)
    cols = ["time"] + canales

    cps_global = None
    multiplicity_counts = np.zeros(n_canales + 1, dtype="int64")
    first_time = None
    last_time = None

    dtype = {c: "float32" for c in canales}

    reader = pd.read_csv(
        ruta_csv,
        usecols=cols,
        chunksize=chunksize,
        dtype=dtype,
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=["time"])
        if chunk.empty:
            continue

        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time"])
        if chunk.empty:
            continue

        tmin = chunk["time"].min()
        tmax = chunk["time"].max()
        if first_time is None or tmin < first_time:
            first_time = tmin
        if last_time is None or tmax > last_time:
            last_time = tmax

        arr = chunk[canales].to_numpy(copy=False)
        active_per_event = np.count_nonzero(arr > 0.0, axis=1).astype(np.int64)
        multiplicity_counts += np.bincount(active_per_event, minlength=n_canales + 1)

        chunk["sec"] = chunk["time"].dt.floor("s")
        grouped = chunk.groupby("sec")[canales].sum()
        cps_global = grouped if cps_global is None else cps_global.add(grouped, fill_value=0)

        del chunk, arr, active_per_event, grouped

    if cps_global is None:
        raise ValueError("No data accumulated. Empty CSV or no valid 'time' column?")

    cps_global = cps_global.sort_index().asfreq("1s", fill_value=0).astype("float32")
    return cps_global, multiplicity_counts, first_time, last_time


def _suavizar_rolling(cps: pd.DataFrame, ventana_seg: int = 5) -> pd.DataFrame:
    return cps.rolling(window=ventana_seg, center=True, min_periods=1).mean()


def _plot_cps_rolling(cps_roll: pd.DataFrame, png_path: Path):
    plt.figure(figsize=(12, 6))
    for col in cps_roll.columns:
        plt.plot(cps_roll.index, cps_roll[col], linewidth=0.8, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Counts per second (rolling mean)")
    plt.title("Per-second counts for channels (rolling)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()


def _hist_desviacion_media(cps: pd.DataFrame, png_path: Path):
    media_por_canal = cps.mean(axis=0)
    media_global = media_por_canal.mean()
    desviacion_pct = (media_por_canal - media_global) / media_global * 100.0

    plt.figure(figsize=(8, 5))
    plt.hist(desviacion_pct.values, bins=10)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Deviation from global mean (%)")
    plt.ylabel("Number of channels")
    plt.title("Channel-to-global mean deviation")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    return desviacion_pct


def _tabla_rendimiento_canal(cps: pd.DataFrame, csv_path: Path):
    media_por_canal = cps.mean(axis=0)
    media_global = media_por_canal.mean()
    total_counts = cps.sum(axis=0)
    std_cps = cps.std(axis=0)
    max_cps = cps.max(axis=0)
    frac_total = total_counts / total_counts.sum()
    desviacion_pct = (media_por_canal - media_global) / media_global * 100.0

    summary = pd.DataFrame(
        {
            "channel": media_por_canal.index,
            "total_counts": total_counts.values,
            "mean_cps": media_por_canal.values,
            "std_cps": std_cps.values,
            "max_cps": max_cps.values,
            "global_mean_cps": media_global,
            "deviation_pct": desviacion_pct.values,
            "fraction_of_total": frac_total.values,
        }
    )
    summary.to_csv(csv_path, index=False)
    return summary


def _hist_multiplicidad(multiplicity_counts: np.ndarray, png_path: Path):
    n_canales = len(multiplicity_counts) - 1
    ks = np.arange(1, n_canales + 1)
    counts = multiplicity_counts[1:]

    if counts.sum() > 0:
        max_k = int(np.max(ks[counts > 0]))
        ks = ks[:max_k]
        counts = counts[:max_k]
    else:
        max_k = n_canales

    plt.figure(figsize=(8, 5))
    plt.bar(ks, counts, align="center")
    plt.xlabel("Number of active channels per event")
    plt.ylabel("Number of events")
    plt.title("Event multiplicity (active channels)")

    if max_k <= 30:
        plt.xticks(ks)
    else:
        step = max(1, max_k // 15)
        plt.xticks(np.arange(1, max_k + 1, step))

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()


def run_rate_analysis(
    in_path: Path,
    out_dir: Path,
    channels_start: int,
    rate_n_channels: int,
    rolling_window_s: int,
    chunksize: int,
    no_plots: bool,
    use_physical_layout: bool,
):
    """Ejecuta analisis de tasa usando la seleccion de canales pedida."""

    header_cols = list(pd.read_csv(in_path, nrows=0).columns)
    num2name = _extract_channels(header_cols)

    if use_physical_layout:
        canales = _select_explicit_channel_numbers(num2name, _default_physical_channel_numbers())
    else:
        canales = _select_contiguous_channels(num2name, channels_start, rate_n_channels)

    base = in_path.stem
    cps, multiplicity_counts, tmin, tmax = _acumular_cps_y_multiplicidad(
        in_path,
        canales=canales,
        chunksize=chunksize,
    )

    print(f"[RATE] Time range in data: {tmin} to {tmax}")
    print(f"[RATE] Total seconds in cps: {len(cps)}")

    cps_roll = _suavizar_rolling(cps, ventana_seg=rolling_window_s)

    # CSV de resumen por canal
    csv_rend = out_dir / f"{base}_channel_performance.csv"
    _tabla_rendimiento_canal(cps, csv_rend)
    print(f"[RATE] Channel performance table: {csv_rend}")

    if no_plots:
        print("[RATE] --no-plots activo: se omiten PNGs de tasa.")
        return

    png_cps = out_dir / f"{base}_cps_rolling.png"
    _plot_cps_rolling(cps_roll, png_cps)
    print(f"[RATE] Saved CPS rolling plot: {png_cps}")

    png_desv = out_dir / f"{base}_desviacion_media.png"
    _hist_desviacion_media(cps, png_desv)
    print(f"[RATE] Saved deviation histogram: {png_desv}")

    png_mult = out_dir / f"{base}_multiplicidad_eventos.png"
    _hist_multiplicidad(multiplicity_counts, png_mult)
    print(f"[RATE] Saved event multiplicity plot: {png_mult}")


# -----------------------------
# Parte B: Coincidencias por activacion (tasaV1)
# -----------------------------

def _pick_engine(engine: str) -> str:
    if engine != "auto":
        return engine
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        return "c"


def _count_data_rows_fast(path: Path, block_size: int = 16 * 1024 * 1024) -> int:
    total_newlines = 0
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            total_newlines += b.count(b"\n")

    if path.stat().st_size == 0:
        return 0

    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)

    if last != b"\n":
        total_newlines += 1

    return max(0, total_newlines - 1)


def _save_heatmap(
    M: np.ndarray,
    xlabels,
    ylabels,
    title: str,
    outpath: Path,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    plt.figure(figsize=(10, 8))
    if center is None:
        im = plt.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    else:
        from matplotlib.colors import TwoSlopeNorm

        finite = np.isfinite(M)
        if not np.any(finite):
            im = plt.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
        else:
            vvmin = np.nanmin(M) if vmin is None else vmin
            vvmax = np.nanmax(M) if vmax is None else vmax
            im = plt.imshow(M, aspect="auto", norm=TwoSlopeNorm(vmin=vvmin, vcenter=center, vmax=vvmax))

    plt.colorbar(im)
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _finite_global_limits(*arrays: np.ndarray) -> tuple[float | None, float | None]:
    vals = []
    for arr in arrays:
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return None, None
    joined = np.concatenate(vals)
    return float(np.min(joined)), float(np.max(joined))


def _symmetric_limits_about(center: float, *arrays: np.ndarray) -> tuple[float | None, float | None]:
    vals = []
    for arr in arrays:
        arr = np.asarray(arr)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return None, None
    joined = np.concatenate(vals)
    radius = float(np.max(np.abs(joined - center)))
    return center - radius, center + radius


def run_coincidence_analysis(
    in_path: Path,
    out_dir: Path,
    chunk_size: int,
    count_lines: bool,
    engine: str,
    channels_start: int,
    no_plots: bool,
    use_physical_layout: bool,
):
    """Ejecuta el analisis de coincidencias (activacion) y guarda todo en out_dir."""

    header_cols = list(pd.read_csv(in_path, nrows=0).columns)
    num2name = _extract_channels(header_cols)

    if use_physical_layout:
        ch_columns = _select_explicit_channel_numbers(num2name, _default_physical_channel_numbers())
        n_ch = len(ch_columns)
        g = 15
        s_p1g1 = slice(0, g)
        s_p1g2 = slice(g, 2 * g)
        s_p2g1 = slice(2 * g, 3 * g)
        s_p2g2 = slice(3 * g, 4 * g)
    else:
        ch_columns = _select_channel_list_for_coincidences(num2name, channels_start)
        n_ch = len(ch_columns)

        # Paneles por POSICION
        if n_ch == 60:
            g = 15
            s_p1g1 = slice(0, g)
            s_p1g2 = slice(g, 2 * g)
            s_p2g1 = slice(2 * g, 3 * g)
            s_p2g2 = slice(3 * g, 4 * g)
        else:
            g = 16
            s_p1g1 = slice(0, g)
            s_p1g2 = slice(g, 2 * g)
            s_p2g1 = slice(2 * g, 3 * g)
            s_p2g2 = slice(3 * g, 4 * g)

    coinc1 = np.zeros((g, g), dtype=np.int64)
    coinc2 = np.zeros((g, g), dtype=np.int64)

    bool_a = np.empty((chunk_size, g), dtype=np.bool_)
    bool_b = np.empty((chunk_size, g), dtype=np.bool_)
    A = np.empty((chunk_size, g), dtype=np.float32)
    B = np.empty((chunk_size, g), dtype=np.float32)

    total_rows = _count_data_rows_fast(in_path) if (count_lines and tqdm is not None) else None
    use_engine = _pick_engine(engine)

    reader = pd.read_csv(
        in_path,
        usecols=ch_columns,
        chunksize=chunk_size,
        engine=use_engine,
        low_memory=False,
        na_filter=False,
        keep_default_na=False,
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_rows, unit="filas", dynamic_ncols=True, desc="[COINC] Procesando")

    rows_done = 0
    for chunk in reader:
        X = chunk.to_numpy(copy=False)
        n = X.shape[0]

        # Panel 1
        np.not_equal(X[:, s_p1g1], 0, out=bool_a[:n, :])
        np.not_equal(X[:, s_p1g2], 0, out=bool_b[:n, :])
        A[:n, :] = bool_a[:n, :]
        B[:n, :] = bool_b[:n, :]
        coinc1 += (A[:n, :].T @ B[:n, :]).astype(np.int64)

        # Panel 2
        np.not_equal(X[:, s_p2g1], 0, out=bool_a[:n, :])
        np.not_equal(X[:, s_p2g2], 0, out=bool_b[:n, :])
        A[:n, :] = bool_a[:n, :]
        B[:n, :] = bool_b[:n, :]
        coinc2 += (A[:n, :].T @ B[:n, :]).astype(np.int64)

        rows_done += n
        if pbar is not None:
            pbar.update(n)
        elif rows_done % (10 * chunk_size) == 0:
            print(f"[COINC] Procesadas ~{rows_done:,} filas...")

    if pbar is not None:
        pbar.close()

    # Matriz completa (rellenando SOLO los bloques físicos)
    C = np.zeros((n_ch, n_ch), dtype=np.int64)
    C[0:g, g:2 * g] = coinc1
    C[g:2 * g, 0:g] = coinc1.T
    C[2 * g:3 * g, 3 * g:4 * g] = coinc2
    C[3 * g:4 * g, 2 * g:3 * g] = coinc2.T

    dfC = pd.DataFrame(C, index=ch_columns, columns=ch_columns)
    dfC.to_csv(out_dir / "coincidencias_activaciones.csv")

    if no_plots:
        print("[COINC] --no-plots activo: se omiten PNGs de coincidencias.")
        return

    panel1_g1 = ch_columns[s_p1g1]
    panel1_g2 = ch_columns[s_p1g2]
    panel2_g1 = ch_columns[s_p2g1]
    panel2_g2 = ch_columns[s_p2g2]

    sub_counts_p1 = C[0:g, g:2 * g]
    sub_counts_p2 = C[2 * g:3 * g, 3 * g:4 * g]
    counts_vmin, counts_vmax = _finite_global_limits(sub_counts_p1, sub_counts_p2)

    _save_heatmap(
        sub_counts_p1,
        panel1_g2,
        panel1_g1,
        f"Coincidences (Panel 1) - {in_path.stem}",
        out_dir / "heatmap_coincidencias_panel1.png",
        vmin=counts_vmin,
        vmax=counts_vmax,
    )
    _save_heatmap(
        sub_counts_p2,
        panel2_g2,
        panel2_g1,
        f"Coincidences (Panel 2) - {in_path.stem}",
        out_dir / "heatmap_coincidencias_panel2.png",
        vmin=counts_vmin,
        vmax=counts_vmax,
    )

    positive = C[C > 0]
    if positive.size == 0:
        print("[COINC] No se encontraron coincidencias > 0. Se omiten métricas log/porcentajes.")
        return

    mean_val = positive.mean()

    variacion_pct = (C - mean_val) / mean_val * 100.0
    variacion_pct[C <= 0] = np.nan
    pd.DataFrame(variacion_pct, index=ch_columns, columns=ch_columns).to_csv(out_dir / "coincidencias_variacion_porcentaje.csv")

    log_counts = np.log10(C.astype(np.float64) + 1.0)
    pd.DataFrame(log_counts, index=ch_columns, columns=ch_columns).to_csv(out_dir / "coincidencias_log10_counts.csv")

    sub_log_p1 = log_counts[0:g, g:2 * g]
    sub_log_p2 = log_counts[2 * g:3 * g, 3 * g:4 * g]
    log_vmin, log_vmax = _finite_global_limits(sub_log_p1, sub_log_p2)

    _save_heatmap(
        sub_log_p1,
        panel1_g2,
        panel1_g1,
        f"log10(count + 1) (Panel 1) - {in_path.stem}",
        out_dir / "heatmap_coincidencias_log_panel1.png",
        vmin=log_vmin,
        vmax=log_vmax,
    )
    _save_heatmap(
        sub_log_p2,
        panel2_g2,
        panel2_g1,
        f"log10(count + 1) (Panel 2) - {in_path.stem}",
        out_dir / "heatmap_coincidencias_log_panel2.png",
        vmin=log_vmin,
        vmax=log_vmax,
    )

    ratio = np.full_like(C, np.nan, dtype=np.float64)
    mask = C > 0
    ratio[mask] = C[mask] / mean_val
    log_ratio = np.full_like(ratio, np.nan, dtype=np.float64)
    log_ratio[mask] = np.log10(ratio[mask])
    pd.DataFrame(log_ratio, index=ch_columns, columns=ch_columns).to_csv(out_dir / "coincidencias_log10_rel_mean.csv")

    sub1 = log_ratio[0:g, g:2 * g]
    sub2 = log_ratio[2 * g:3 * g, 3 * g:4 * g]
    rel_vmin, rel_vmax = _symmetric_limits_about(0.0, sub1, sub2)

    if np.isfinite(sub1).any():
        _save_heatmap(
            sub1,
            panel1_g2,
            panel1_g1,
            f"log10(C / mean) (Panel 1) - {in_path.stem}",
            out_dir / "heatmap_log10_rel_mean_panel1.png",
            center=0.0,
            vmin=rel_vmin,
            vmax=rel_vmax,
        )

    if np.isfinite(sub2).any():
        _save_heatmap(
            sub2,
            panel2_g2,
            panel2_g1,
            f"log10(C / mean) (Panel 2) - {in_path.stem}",
            out_dir / "heatmap_log10_rel_mean_panel2.png",
            center=0.0,
            vmin=rel_vmin,
            vmax=rel_vmax,
        )


# -----------------------------
# CLI
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analisis unificado: tasa/estabilidad por canal + coincidencias por activacion. "
            "Todas las salidas se guardan en <folder_csv>/graficas_<stem>/"
        )
    )
    ap.add_argument("archivo", help="CSV de entrada")

    # Controles generales
    ap.add_argument("--only", choices=["all", "rate", "coinc"], default="all", help="Ejecutar solo una parte")
    ap.add_argument("--no-plots", action="store_true", help="No generar PNGs (solo CSVs)")
    ap.add_argument(
        "--use-physical-layout",
        action="store_true",
        default=True,
        help="Usa P1=ch1..ch30 y P2=ch32..ch61 (por defecto activo)",
    )
    ap.add_argument(
        "--no-physical-layout",
        dest="use_physical_layout",
        action="store_false",
        help="Vuelve al esquema contiguo original",
    )

    # Parametros de RATE
    ap.add_argument("--rate-chunksize", type=int, default=200_000, help="Chunk size para tasa (streaming)")
    ap.add_argument("--ventana-rolling", type=int, default=5, help="Rolling window en segundos (tasa)")
    ap.add_argument(
        "--rate-channels-start",
        type=int,
        default=1,
        help="Canal inicial para tasa. Solo se usa si activas --no-physical-layout",
    )
    ap.add_argument(
        "--rate-n-channels",
        type=int,
        default=60,
        help="Numero de canales para tasa. Solo se usa si activas --no-physical-layout",
    )

    # Parametros de COINC
    ap.add_argument("--chunk-size", type=int, default=300_000, help="Chunk size para coincidencias")
    ap.add_argument("--count-lines", action="store_true", help="Progreso % exacto (cuenta filas primero).")
    ap.add_argument("--engine", choices=["auto", "c", "pyarrow"], default="c")
    ap.add_argument("--channels-start", type=int, default=1, help="1 si tus canales son ch1..; 0 si son ch00..")

    args = ap.parse_args()

    in_path = Path(args.archivo).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}")

    out_dir = in_path.parent / f"graficas_{in_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Salidas en: {out_dir}")

    if args.only in ("all", "rate"):
        run_rate_analysis(
            in_path=in_path,
            out_dir=out_dir,
            channels_start=args.rate_channels_start,
            rate_n_channels=args.rate_n_channels,
            rolling_window_s=args.ventana_rolling,
            chunksize=args.rate_chunksize,
            no_plots=args.no_plots,
            use_physical_layout=args.use_physical_layout,
        )

    if args.only in ("all", "coinc"):
        run_coincidence_analysis(
            in_path=in_path,
            out_dir=out_dir,
            chunk_size=args.chunk_size,
            count_lines=args.count_lines,
            engine=args.engine,
            channels_start=args.channels_start,
            no_plots=args.no_plots,
            use_physical_layout=args.use_physical_layout,
        )

    print(f"Listo. Salidas en: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
