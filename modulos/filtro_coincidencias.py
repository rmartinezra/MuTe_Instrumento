#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filtro_coincidencias_v2.py

Cambios clave respecto a la versión anterior:
1) HIT = (valor == 1) (no "!= 0"). En la versión previa, cualquier valor no-cero contaba como hit. fileciteturn3file3L7-L13
2) Por defecto: elimina (drop) cualquier fila donde algún canal tenga valor distinto de {0,1},
   DESPUÉS de aplicar el "zeroing" opcional.
3) Si pasas --zero-ch, esos canales se fuerzan a 0 y además se escriben como 0 en el CSV de salida
   (antes solo se usaban para el filtro pero la salida quedaba con los valores originales).
4) Binariza/normaliza: convierte canales a numérico y rellena NaN con 0, y eso mismo es lo que se escribe.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# Grupos por planos: 4 grupos de 15 (ajusta si tu mapping es otro)
G1 = list(range(1, 16))
G2 = list(range(16, 31))
G3 = list(range(32, 47))
G4 = list(range(47, 62))


def _parse_int_list_csv(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _ch_name(n: int) -> str:
    return f"ch{n:02d}"


def _infer_zero_positions(ch_to_pos: dict[int, int], zero_channels: list[int]):
    """
    Mapea números provistos a posiciones de columnas.
    Regla:
      - Si existe chN, usa N.
      - Si NO existe chN pero existe ch(N-1), sugiere 0-index (warn) y usa N-1.
    """
    zero_pos = []
    used = []
    missing = []
    aliased = []
    for n in zero_channels:
        if n in ch_to_pos:
            zero_pos.append(ch_to_pos[n])
            used.append(n)
        elif (n - 1) in ch_to_pos and n not in ch_to_pos:
            zero_pos.append(ch_to_pos[n - 1])
            used.append(n - 1)
            aliased.append((n, n - 1))
        else:
            missing.append(n)
    return np.array(zero_pos, dtype=int) if zero_pos else np.array([], dtype=int), used, missing, aliased


def parse_args():
    p = argparse.ArgumentParser(description="Filtra coincidencias 2-fold y 4-fold con validación binaria (0/1).")
    p.add_argument("archivo", help="CSV de entrada.")
    p.add_argument("--time-col", default="time", help="Columna temporal (defecto: time).")
    p.add_argument("--chunk-size", type=int, default=300_000, help="Filas por chunk (defecto: 300000).")
    p.add_argument("--outdir", default=None, help="Directorio de salida (defecto: mismo dir del input).")

    # NUEVO (opcional): canales a forzar a 0
    p.add_argument(
        "--zero-ch",
        default="",
        help="Canales a forzar a 0 SIEMPRE (ej: '7,59'). Si se omite, no hace zeroing.",
    )

    p.add_argument("--write-coinc2", action="store_true", help="Genera además salida coinc2 (exactamente 2 hits).")
    p.add_argument("--strict-4fold", action="store_true", help="Además exige exactamente 4 hits en ch01..ch60.")
    p.add_argument("--keep-all-cols", action="store_true", help="Escribe todas las columnas originales (no recomendado).")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.archivo).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else in_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(in_path, nrows=0).columns.tolist()
    if args.time_col not in header:
        raise ValueError(f"No encontré la columna '{args.time_col}' en el CSV.")

    ch_cols = [c for c in header if re.fullmatch(r"ch\d+", c)]
    if not ch_cols:
        raise ValueError("No encontré columnas tipo chNN.")
    ch_cols = sorted(ch_cols, key=lambda x: int(x[2:]))

    # Lectura: por defecto solo time + ch*
    if args.keep_all_cols:
        usecols = header
        out_cols = header
    else:
        usecols = [args.time_col] + ch_cols
        out_cols = usecols

    # Mapa numérico de canal -> posición en ch_cols
    ch_to_pos = {int(c[2:]): i for i, c in enumerate(ch_cols)}

    def idx_list(ch_numbers):
        missing = [n for n in ch_numbers if n not in ch_to_pos]
        if missing:
            raise ValueError(f"Faltan columnas para estos canales: {missing}")
        return np.array([ch_to_pos[n] for n in ch_numbers], dtype=int)

    g1_pos = idx_list(G1)
    g2_pos = idx_list(G2)
    g3_pos = idx_list(G3)
    g4_pos = idx_list(G4)
    g_all_pos = idx_list(list(range(1, 61)))  # para strict_4fold

    # Zeroing opcional
    zero_channels = _parse_int_list_csv(args.zero_ch)
    zero_pos, zero_used, zero_missing, zero_aliased = _infer_zero_positions(ch_to_pos, zero_channels)

    # Salidas
    stem = in_path.stem
    out_coinc4 = outdir / f"{stem}_coinc4.csv"
    out_coinc2 = outdir / f"{stem}_coinc2.csv" if args.write_coinc2 else None
    out_report = outdir / f"{stem}_filter_report.txt"

    # Limpia
    out_coinc4.write_text("")
    if out_coinc2:
        out_coinc2.write_text("")

    wrote4 = False
    wrote2 = False

    reader = pd.read_csv(in_path, usecols=usecols, chunksize=args.chunk_size)
    iterator = tqdm(reader, desc="Filtrando", unit="chunk", dynamic_ncols=True) if tqdm else reader

    total_rows = 0
    dropped_nonbinary = 0
    kept2 = 0
    kept4 = 0

    for chunk in iterator:
        total_rows += len(chunk)

        # Normaliza canales a numérico (esto también se escribe a la salida)
        chunk[ch_cols] = chunk[ch_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int16)

        # Aplica zeroing (esto modifica la salida también)
        if zero_pos.size > 0:
            # zero_pos es posición en ch_cols; lo aplicamos por nombre
            zero_names = [ch_cols[i] for i in zero_pos.tolist()]
            chunk.loc[:, zero_names] = 0

        # DROP por defecto: cualquier valor != 0 y != 1 en cualquier canal
        X = chunk[ch_cols].to_numpy(dtype=np.int16)
        nonbinary = (X != 0) & (X != 1)
        m_ok = ~np.any(nonbinary, axis=1)
        if not np.all(m_ok):
            dropped_nonbinary += int((~m_ok).sum())
            chunk = chunk.loc[m_ok].copy()
            if len(chunk) == 0:
                continue
            X = chunk[ch_cols].to_numpy(dtype=np.int16)

        # HIT: valor EXACTO 1
        H = (X == 1)

        # coinc2: exactamente 2 hits en todo el bloque ch*
        if out_coinc2 is not None:
            c_all = H.sum(axis=1)
            m2 = (c_all == 2)
            if np.any(m2):
                sub2 = chunk.loc[m2, out_cols]
                sub2.to_csv(out_coinc2, mode="a", header=(not wrote2), index=False)
                wrote2 = True
                kept2 += int(m2.sum())

        # coinc4: 1 hit por grupo
        c1 = H[:, g1_pos].sum(axis=1)
        c2 = H[:, g2_pos].sum(axis=1)
        c3 = H[:, g3_pos].sum(axis=1)
        c4 = H[:, g4_pos].sum(axis=1)
        m4 = (c1 == 1) & (c2 == 1) & (c3 == 1) & (c4 == 1)

        if args.strict_4fold:
            c60 = H[:, g_all_pos].sum(axis=1)
            m4 = m4 & (c60 == 4)

        if np.any(m4):
            sub4 = chunk.loc[m4, out_cols]
            sub4.to_csv(out_coinc4, mode="a", header=(not wrote4), index=False)
            wrote4 = True
            kept4 += int(m4.sum())

    # Reporte
    lines = []
    lines.append(f"Input: {in_path}")
    lines.append(f"Rows read: {total_rows}")
    lines.append(f"Rows dropped (non-binary !=0/1) AFTER zeroing: {dropped_nonbinary}")
    if zero_channels:
        lines.append(f"Requested --zero-ch: {zero_channels}")
        if zero_aliased:
            lines.append("NOTE: algunos canales parecían 0-indexed. Alias aplicado:")
            for a,b in zero_aliased:
                lines.append(f"  requested {a} -> used {b} (porque ch{a:02d} no existe y ch{b:02d} sí)")
        if zero_missing:
            lines.append(f"WARNING: canales en --zero-ch no existen y se ignoraron: {zero_missing}")
        lines.append(f"Zeroed channels used: {sorted(set(zero_used))}")
    else:
        lines.append("Zeroing: OFF (no --zero-ch provided)")

    if out_coinc2:
        lines.append(f"Coinc2 kept: {kept2} -> {out_coinc2}")
    else:
        lines.append("Coinc2: OFF (use --write-coinc2 to enable)")

    lines.append(f"Coinc4 kept: {kept4} -> {out_coinc4}")
    lines.append("Hit definition: (value == 1) only.")
    lines.append(f"strict_4fold: {'ON' if args.strict_4fold else 'OFF'}")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
