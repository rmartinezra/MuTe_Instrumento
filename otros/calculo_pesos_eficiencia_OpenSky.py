#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calcula pesos de corrección de eficiencia por píxel desde una matriz de activaciones,
usando la convención física explícita de canales:

  Panel 1 abajo:  X1 = ch01..ch15, Y1 = ch16..ch30
  Panel 2 arriba: X2 = ch32..ch46, Y2 = ch47..ch61   (ch31 sin uso)

Este script evita asumir a ciegas que la matriz ya está "compactada" a 60x60.
Intenta detectar el layout y extraer los bloques correctos.

Admite dos casos:
1) Matriz con etiquetas de filas/columnas tipo ch01..ch61
2) Matriz numérica sin etiquetas:
   - compact60: orden activo compactado [ch01..ch30, ch32..ch61]
   - absolute61: orden absoluto [ch01..ch61], con ch31 presente

Método:
- extrae mapas 15x15:
    P1 = coincidencias X1×Y1
    P2 = coincidencias X2×Y2
- ajusta una superficie cuadrática 2D sobre log(mapa)
- respuesta relativa = mapa / ajuste
- normaliza a media 1
- peso = 1 / respuesta_relativa_norm

Uso:
python3 compute_efficiency_weights_v2.py coincidencias_activaciones_OpenSky.csv --outdir weights_out
python3 compute_efficiency_weights_v2.py coincidencias_activaciones_OpenSky.csv --layout compact60 --outdir weights_out
python3 compute_efficiency_weights_v2.py coincidencias_activaciones_OpenSky.csv --layout absolute61 --outdir weights_out
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PHYS_X1 = [f"ch{i:02d}" for i in range(1, 16)]
PHYS_Y1 = [f"ch{i:02d}" for i in range(16, 31)]
PHYS_X2 = [f"ch{i:02d}" for i in range(32, 47)]
PHYS_Y2 = [f"ch{i:02d}" for i in range(47, 62)]

COMPACT60 = [f"ch{i:02d}" for i in range(1, 31)] + [f"ch{i:02d}" for i in range(32, 62)]
ABSOLUTE61 = [f"ch{i:02d}" for i in range(1, 62)]


def polyfit2d_quadratic_log_surface(grid15: np.ndarray) -> np.ndarray:
    z = np.asarray(grid15, dtype=float)
    if z.shape != (15, 15):
        raise ValueError(f"Se esperaba una matriz 15x15, llegó {z.shape}")

    x, y = np.meshgrid(np.arange(15), np.arange(15), indexing="ij")
    mask = z > 0
    if mask.sum() < 6:
        raise ValueError("No hay suficientes celdas positivas para ajustar la superficie.")

    X = np.column_stack([
        np.ones(mask.sum()),
        x[mask], y[mask],
        x[mask] ** 2, x[mask] * y[mask], y[mask] ** 2,
    ])
    target = np.log(z[mask])
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)

    Xfull = np.column_stack([
        np.ones(z.size),
        x.ravel(), y.ravel(),
        x.ravel() ** 2, x.ravel() * y.ravel(), y.ravel() ** 2,
    ])
    fit = np.exp((Xfull @ coef).reshape(15, 15))
    return fit


def normalize_channel_name(x):
    s = str(x).strip()
    if s.lower().startswith("ch"):
        tail = s[2:]
        if tail.isdigit():
            return f"ch{int(tail):02d}"
    if s.isdigit():
        return f"ch{int(s):02d}"
    return s


def try_read_labeled_matrix(path: Path):
    raw = pd.read_csv(path, index_col=0)
    raw.index = [normalize_channel_name(v) for v in raw.index]
    raw.columns = [normalize_channel_name(v) for v in raw.columns]
    common = set(raw.index) & set(raw.columns)
    if all(ch in common for ch in PHYS_X1 + PHYS_Y1 + PHYS_X2 + PHYS_Y2):
        return raw.apply(pd.to_numeric, errors="coerce")
    return None


def read_numeric_matrix(path: Path) -> np.ndarray:
    raw = pd.read_csv(path, header=None)
    mat = raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isfinite(mat).sum() == 0:
        mat = raw.apply(pd.to_numeric, errors="coerce").to_numpy()
    return mat.astype(float)


def detect_numeric_layout(mat: np.ndarray, forced_layout: str | None):
    nrows, ncols = mat.shape
    if forced_layout is not None:
        return forced_layout
    if nrows >= 61 and ncols >= 61:
        return "absolute61"
    if nrows >= 60 and ncols >= 60:
        return "compact60"
    raise ValueError(f"No puedo inferir el layout desde una matriz {mat.shape}")


def extract_from_labeled(df: pd.DataFrame):
    p1 = df.loc[PHYS_X1, PHYS_Y1].to_numpy(float)
    p2 = df.loc[PHYS_X2, PHYS_Y2].to_numpy(float)
    return p1, p2, "labeled"


def extract_from_numeric(mat: np.ndarray, layout: str):
    if layout == "compact60":
        compact = COMPACT60
        index = {ch: i for i, ch in enumerate(compact)}
    elif layout == "absolute61":
        index = {ch: i for i, ch in enumerate(ABSOLUTE61)}
    else:
        raise ValueError(f"layout no soportado: {layout}")

    ix_x1 = [index[ch] for ch in PHYS_X1]
    ix_y1 = [index[ch] for ch in PHYS_Y1]
    ix_x2 = [index[ch] for ch in PHYS_X2]
    ix_y2 = [index[ch] for ch in PHYS_Y2]

    p1 = mat[np.ix_(ix_x1, ix_y1)]
    p2 = mat[np.ix_(ix_x2, ix_y2)]
    return p1.astype(float), p2.astype(float), layout


def compute_weights_from_maps(p1: np.ndarray, p2: np.ndarray):
    fit1 = polyfit2d_quadratic_log_surface(p1)
    fit2 = polyfit2d_quadratic_log_surface(p2)

    rel1 = p1 / fit1
    rel2 = p2 / fit2

    rel1 = rel1 / np.nanmean(rel1)
    rel2 = rel2 / np.nanmean(rel2)

    w1 = 1.0 / rel1
    w2 = 1.0 / rel2
    return fit1, fit2, rel1, rel2, w1, w2


def save_matrix(arr: np.ndarray, path: Path, index_name: str = "x_index", col_prefix: str = "y"):
    df = pd.DataFrame(arr, index=np.arange(arr.shape[0]), columns=[f"{col_prefix}{j:02d}" for j in range(arr.shape[1])])
    df.index.name = index_name
    df.to_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Calcula pesos de eficiencia por píxel desde activaciones Open-Sky.")
    parser.add_argument("activations_csv", type=Path, help="Archivo de activaciones")
    parser.add_argument("--outdir", type=Path, default=Path("efficiency_weights_out"))
    parser.add_argument("--layout", choices=["compact60", "absolute61"], default=None,
                        help="Forzar layout numérico si el archivo no tiene etiquetas.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    labeled = try_read_labeled_matrix(args.activations_csv)
    if labeled is not None:
        p1, p2, layout_used = extract_from_labeled(labeled)
    else:
        mat = read_numeric_matrix(args.activations_csv)
        layout_used = detect_numeric_layout(mat, args.layout)
        p1, p2, layout_used = extract_from_numeric(mat, layout_used)

    fit1, fit2, rel1, rel2, w1, w2 = compute_weights_from_maps(p1, p2)

    save_matrix(p1,   args.outdir / "panel1_activation.csv")
    save_matrix(p2,   args.outdir / "panel2_activation.csv")
    save_matrix(fit1, args.outdir / "panel1_fit.csv")
    save_matrix(fit2, args.outdir / "panel2_fit.csv")
    save_matrix(rel1, args.outdir / "panel1_relative_response.csv")
    save_matrix(rel2, args.outdir / "panel2_relative_response.csv")
    save_matrix(w1,   args.outdir / "panel1_weights.csv")
    save_matrix(w2,   args.outdir / "panel2_weights.csv")

    summary = []
    summary.append("Pesos de eficiencia calculados desde archivo de activaciones.")
    summary.append("")
    summary.append("Convención física usada:")
    summary.append("  Panel 1 abajo:  X1 = ch01..ch15, Y1 = ch16..ch30")
    summary.append("  Panel 2 arriba: X2 = ch32..ch46, Y2 = ch47..ch61  (ch31 sin uso)")
    summary.append("")
    summary.append(f"Layout detectado/usado: {layout_used}")
    summary.append("")
    summary.append("Definición:")
    summary.append("  fit = superficie cuadrática 2D ajustada a log(mapa)")
    summary.append("  rel = mapa / fit")
    summary.append("  rel_norm = rel / mean(rel)")
    summary.append("  w = 1 / rel_norm")
    summary.append("")
    summary.append("Resumen numérico:")
    summary.append(f"  Panel 1 activation sum = {np.nansum(p1):.6f}")
    summary.append(f"  Panel 2 activation sum = {np.nansum(p2):.6f}")
    summary.append(f"  Panel 1 weight min/max = {np.nanmin(w1):.6f} / {np.nanmax(w1):.6f}")
    summary.append(f"  Panel 2 weight min/max = {np.nanmin(w2):.6f} / {np.nanmax(w2):.6f}")
    summary.append(f"  Panel 1 weight mean    = {np.nanmean(w1):.6f}")
    summary.append(f"  Panel 2 weight mean    = {np.nanmean(w2):.6f}")

    (args.outdir / "weights_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"Listo. Archivos guardados en: {args.outdir}")
    print(f"Layout usado: {layout_used}")


if __name__ == "__main__":
    main()
