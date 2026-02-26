#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

CH_RE = re.compile(r"^ch(\d+)$", re.IGNORECASE)
AREA_DEFAULT = 0.36


def count_data_rows_fast(path: Path, block_size: int = 16 * 1024 * 1024) -> int:
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
        if f.read(1) != b"\n":
            total_newlines += 1
    return max(0, total_newlines - 1)


def parse_channels_from_header(cols: list[str]) -> dict[int, str]:
    num2name = {}
    for c in cols:
        m = CH_RE.match(str(c).strip())
        if m:
            n = int(m.group(1))
            num2name.setdefault(n, c)
    return num2name


def choose_channel_block(num2name: dict[int, str], channels_start: int) -> list[str]:
    if all((channels_start + k) in num2name for k in range(60)):
        n = 60
    elif all((channels_start + k) in num2name for k in range(64)):
        n = 64
    else:
        missing = [channels_start + k for k in range(60) if (channels_start + k) not in num2name][:20]
        raise ValueError(f"No encuentro bloque contiguo desde ch{channels_start}. Faltan (ej): {missing}")
    return [num2name[channels_start + k] for k in range(n)]


def rolling_median_std_fast(x: np.ndarray, window: int):
    try:
        import bottleneck as bn
        med = bn.move_median(x, window=window, min_count=window)
        std = bn.move_std(x, window=window, min_count=window, ddof=1)
        return med, std
    except Exception:
        import pandas as pd
        s = pd.Series(x)
        return s.rolling(window).median().to_numpy(), s.rolling(window).std().to_numpy()


def backend_polars(csv_path: Path, time_col: str, ch_cols: list[str], window: int, area: float,
                   polars_threads: int | None):
    # IMPORTANTE: setear hilos ANTES de importar polars (más fiable entre versiones)
    if polars_threads is not None and polars_threads > 0:
        os.environ["POLARS_MAX_THREADS"] = str(polars_threads)

    import polars as pl

    # scan_csv compatible: nada de columns=
    try:
        lf = pl.scan_csv(str(csv_path), try_parse_dates=True)
    except TypeError:
        lf = pl.scan_csv(str(csv_path))

    lf = lf.select([time_col] + ch_cols)

    # Asegurar datetime y truncar a segundo
    lf = lf.with_columns(
        pl.col(time_col).cast(pl.Datetime, strict=False).alias(time_col)
    ).drop_nulls([time_col])

    lf = lf.with_columns(
        pl.col(time_col).dt.truncate("1s").alias("t")
    )

    # Suma por segundo (aditiva)
    agg_exprs = [pl.col(c).sum().alias(c) for c in ch_cols]
    per_sec = (
        lf.group_by("t")
          .agg(agg_exprs)
          .sort("t")
    )

    # Paneles por posición: primera mitad / segunda mitad
    half = len(ch_cols) // 2
    p1 = ch_cols[:half]
    p2 = ch_cols[half:]

    per_sec = per_sec.with_columns([
        pl.sum_horizontal([pl.col(c) for c in p1]).alias("p1"),
        pl.sum_horizontal([pl.col(c) for c in p2]).alias("p2"),
    ])

    # Rolling para ambos paneles
    per_sec = per_sec.with_columns([
        pl.col("p1").rolling_median(window_size=window).alias("med1"),
        pl.col("p1").rolling_std(window_size=window).alias("std1"),
        pl.col("p2").rolling_median(window_size=window).alias("med2"),
        pl.col("p2").rolling_std(window_size=window).alias("std2"),
    ])

    out = per_sec.select(["t", "med1", "std1", "med2", "std2"]).collect(streaming=True)

    t = out["t"].to_numpy()
    med1 = out["med1"].to_numpy()
    std1 = out["std1"].to_numpy()
    med2 = out["med2"].to_numpy()
    std2 = out["std2"].to_numpy()

    y1 = med1 / area
    lo1 = (med1 - std1 / 2.0) / area
    hi1 = (med1 + std1 / 2.0) / area

    y2 = med2 / area
    lo2 = (med2 - std2 / 2.0) / area
    hi2 = (med2 + std2 / 2.0) / area

    return t, y1, lo1, hi1, y2, lo2, hi2


def backend_pandas(csv_path: Path, time_col: str, ch_cols: list[str], window: int, area: float,
                   chunk_size: int, count_lines: bool, engine: str):
    import pandas as pd

    # pyarrow no soporta chunksize -> forzar 'c'
    if engine == "pyarrow":
        engine = "c"

    total = count_data_rows_fast(csv_path) if (count_lines and tqdm is not None) else None
    pbar = tqdm(total=total, unit="filas", dynamic_ncols=True, desc="Leyendo/agrupando") if tqdm else None

    parts = []
    reader = pd.read_csv(
        csv_path,
        usecols=[time_col] + ch_cols,
        chunksize=chunk_size,
        engine=engine,
        low_memory=False,
    )

    for chunk in reader:
        t = pd.to_datetime(chunk[time_col], errors="coerce")
        m = t.notna()
        if m.any():
            chunk = chunk.loc[m]
            t = t.loc[m].values.astype("datetime64[s]")
            chunk[time_col] = t
            parts.append(chunk.groupby(time_col, sort=False)[ch_cols].sum())

        if pbar:
            pbar.update(len(chunk) if hasattr(chunk, "__len__") else 0)

    if pbar:
        pbar.close()

    if not parts:
        raise RuntimeError("No quedó ningún dato válido tras parsear 'time'.")

    per_sec = pd.concat(parts).groupby(level=0).sum().sort_index()

    half = len(ch_cols) // 2
    p1 = ch_cols[:half]
    p2 = ch_cols[half:]

    # Suma por panel
    p1_sum = per_sec[p1].sum(axis=1).to_numpy(dtype=np.float64)
    p2_sum = per_sec[p2].sum(axis=1).to_numpy(dtype=np.float64)

    # Rolling para ambos paneles
    med1, std1 = rolling_median_std_fast(p1_sum, window)
    med2, std2 = rolling_median_std_fast(p2_sum, window)

    t = per_sec.index.to_numpy()

    y1 = med1 / area
    lo1 = (med1 - std1 / 2.0) / area
    hi1 = (med1 + std1 / 2.0) / area

    y2 = med2 / area
    lo2 = (med2 - std2 / 2.0) / area
    hi2 = (med2 + std2 / 2.0) / area

    return t, y1, lo1, hi1, y2, lo2, hi2


def main():
    ap = argparse.ArgumentParser(description="Rolling median optimizado (Paneles 1 y 2).")
    ap.add_argument("archivo")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--area", type=float, default=AREA_DEFAULT)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--channels-start", type=int, default=1, help="1 si tus canales son ch1..ch60; 0 si son ch00..ch59")
    ap.add_argument("--backend", choices=["auto", "polars", "pandas"], default="auto")
    ap.add_argument("--chunk-size", type=int, default=300_000)
    ap.add_argument("--count-lines", action="store_true")
    ap.add_argument("--engine", choices=["c", "pyarrow"], default="c", help="pandas: usa 'c' para chunks")
    ap.add_argument("--polars-threads", type=int, default=0, help="0=auto; >0 fija hilos en polars")
    args = ap.parse_args()

    csv_path = Path(args.archivo).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Leer header con pandas (solo cabecera)
    import pandas as pd
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    num2name = parse_channels_from_header(header)
    ch_cols = choose_channel_block(num2name, args.channels_start)

    backend = args.backend
    if backend == "auto":
        try:
            import polars as _  # noqa: F401
            backend = "polars"
        except Exception:
            backend = "pandas"

    if backend == "polars":
        print("Usando Polars (multinúcleo). Nota: no hay barra fina de progreso, pero es el más rápido.")
        t, y1, lo1, hi1, y2, lo2, hi2 = backend_polars(
            csv_path, args.time_col, ch_cols, args.window, args.area,
            (args.polars_threads if args.polars_threads > 0 else None),
        )
    else:
        print("Usando Pandas (con progreso).")
        t, y1, lo1, hi1, y2, lo2, hi2 = backend_pandas(
            csv_path, args.time_col, ch_cols, args.window, args.area,
            args.chunk_size, args.count_lines, args.engine,
        )

    out_dir = csv_path.parent / f"graficas_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 10))
    # Panel 1
    plt.plot(t, y1, linestyle="-", label="Panel 1 (rolling median)")
    plt.fill_between(t, lo1, hi1, alpha=0.35)
    # Panel 2
    plt.plot(t, y2, linestyle="-", label="Panel 2 (rolling median)")
    plt.fill_between(t, lo2, hi2, alpha=0.35)

    plt.title("Flujo Paneles 1 y 2")
    plt.xlabel("Tiempo")
    plt.ylabel("Cnts/s*m^2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = out_dir / "rolling_median.png"
    plt.savefig(out_png, dpi=120)
    plt.close()

    print(f"Listo. Guardado en: {out_png}")


if __name__ == "__main__":
    main()
