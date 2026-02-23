#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convertidor CAEN Janus/FERS (formato "shaping": lista de hits por trigger)
a formato MuTe (wide): columna 'time' + ch00..ch63.

Entrada (shaping típico):
  //Start_Time_Epoch:1771532477098
  TStamp_us,Trg_Id,...,CH_Id,...,PHA_HG
  129881.816,0,...,4,...
  129881.816,0,...,18,...
  ...

Salida (MuTe típico):
  time,ch00,ch01,...,ch63
  2026-02-19 20:21:17.227881816,0,0,...,1
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_shaping_header(path: Path) -> dict:
    """Lee metadata tipo //Key:Value al inicio del archivo."""
    meta = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            if not line.startswith("//"):
                break
            m = re.match(r"//([^:]+):(.+)", line.strip())
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
    return meta


def find_first_data_line_idx(path: Path) -> int:
    """Encuentra el índice (0-based) de la primera línea NO comentada (header CSV real)."""
    with path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip() and not line.startswith("//"):
                return i
    raise ValueError("No se encontró una cabecera CSV válida (líneas no comentadas).")


def read_shaping_dataframe(path: Path) -> pd.DataFrame:
    """Lee el CSV saltando el bloque de comentarios //..."""
    header_idx = find_first_data_line_idx(path)
    return pd.read_csv(path, skiprows=header_idx)


def shaping_to_mute(df: pd.DataFrame, start_epoch_ms: int) -> pd.DataFrame:
    """
    Convierte hits por trigger a wide.
    - 1 fila por Trg_Id
    - time = Start_Time_Epoch + TStamp_us (us) en UTC (se guarda naive)
    - chXX = conteo de hits en ese canal dentro del trigger
    """
    required = {"TStamp_us", "Trg_Id", "CH_Id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en shaping: {sorted(missing)}")

    base = pd.Timestamp(int(start_epoch_ms), unit="ms", tz="UTC")

    # timestamp por trigger (primer TStamp_us del grupo)
    t_per = df.groupby("Trg_Id")["TStamp_us"].first()
    times = base + pd.to_timedelta(t_per.values, unit="us")

    out = pd.DataFrame({"time": times.tz_convert(None).astype("datetime64[ns]")})
    out.index = t_per.index  # index = Trg_Id

    # crea columnas ch00..ch63
    for ch in range(64):
        out[f"ch{ch:02d}"] = 0

    # conteos por (trigger, canal)
    counts = df.groupby(["Trg_Id", "CH_Id"]).size()

    for (trg, ch), c in counts.items():
        ch = int(ch)
        if 0 <= ch < 64:
            out.loc[trg, f"ch{ch:02d}"] = int(c)

    out = out.sort_index().reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convertir CSV shaping (hits por trigger) a formato MuTe (wide ch00..ch63)."
    )
    ap.add_argument("input", type=str, help="Ruta al CSV shaping (con comentarios //...).")
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Ruta de salida. Si no se da, se crea '<input>_as_MuTe.csv'.",
    )
    ap.add_argument(
        "--start-epoch-ms",
        type=int,
        default=None,
        help="Sobrescribe Start_Time_Epoch (ms) si no está en el header.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}")

    meta = parse_shaping_header(in_path)
    epoch_ms = args.start_epoch_ms
    if epoch_ms is None:
        if "Start_Time_Epoch" not in meta:
            raise ValueError(
                "No encontré //Start_Time_Epoch en el header y no pasaste --start-epoch-ms."
            )
        epoch_ms = int(meta["Start_Time_Epoch"])

    df = read_shaping_dataframe(in_path)
    out = shaping_to_mute(df, epoch_ms)

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_as_MuTe.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] Escribí: {out_path}  (rows={len(out)}, cols={len(out.columns)})")


if __name__ == "__main__":
    main()
