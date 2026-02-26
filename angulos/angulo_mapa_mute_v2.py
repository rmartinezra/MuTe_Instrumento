
#!/usr/bin/env python3
"""
angulo_mapa_mute_v2.py

Construye:
  (1) mapa N(Δx,Δy) en píxeles (Δ en barras)
  (2) mapa N(θx,θy) (ángulos observados por geometría de barras)
  (3) histograma N(θ) (cenital)
y (opcional) compara un dataset "con plomo" contra un dataset "sin plomo" para
buscar atenuación en un rango de barras del panel superior (por defecto Y2: ch46-53).

Supuestos (para tus CSV tipo "time,ch00..ch63"):
  - 15 barras por plano
  - canales útiles ch01..ch60 (ch00 y ch61..ch63 suelen ser 0)
  - orden:
      X1: ch01..ch15
      Y1: ch16..ch30
      X2: ch31..ch45
      Y2: ch46..ch60   (panel 2 = superior)
  - Δx = x2 - x1, Δy = y2 - y1  (en índices físicos de barra)

Notas importantes:
  - Si un plano tiene multi-hit, por defecto usa centroid (índice promedio ponderado).
  - El "flip" corrige inversión de cableado (índice i -> 14-i). Aquí, por defecto,
    se ofrece --flip-y2 porque en tus datos sin plomo esa opción vuelve Y2 casi simétrico.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd


NB = 15


def _load_matrix(csv_path: str, ch_offset: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    cols = [f"ch{i:02d}" for i in range(ch_offset, ch_offset + 60)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {missing[:5]} ...")
    M = df[cols].to_numpy(int)

    x1 = M[:, 0:NB]
    y1 = M[:, NB:2*NB]
    x2 = M[:, 2*NB:3*NB]
    y2 = M[:, 3*NB:4*NB]
    return x1, y1, x2, y2


def _centroid_idx(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = S.sum(axis=1)
    ok = c >= 1
    inds = np.arange(S.shape[1])[None, :]
    idx = np.rint((S * inds).sum(axis=1) / np.maximum(c, 1)).astype(int)
    idx = np.clip(idx, 0, S.shape[1]-1)
    return idx, ok, c


def _strict_idx(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = S.sum(axis=1)
    ok = c == 1
    idx = np.argmax(S, axis=1).astype(int)
    return idx, ok, c


def _indices_for_events(csv_path: str, mode: str, ch_offset: int,
                        flip_x2: bool, flip_y2: bool,
                        require4: bool = True):
    x1, y1, x2, y2 = _load_matrix(csv_path, ch_offset=ch_offset)

    f = _centroid_idx if mode == "centroid" else _strict_idx

    ix1, okx1, _ = f(x1)
    iy1, oky1, _ = f(y1)
    ix2, okx2, _ = f(x2)
    iy2, oky2, _ = f(y2)

    if flip_x2:
        ix2 = (NB-1) - ix2
    if flip_y2:
        iy2 = (NB-1) - iy2

    ok = (okx1 & oky1 & okx2 & oky2) if require4 else oky2

    return ix1[ok], iy1[ok], ix2[ok], iy2[ok], ok.mean()


def _delta_maps(ix1, iy1, ix2, iy2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = ix2 - ix1
    dy = iy2 - iy1
    # bins de -14..14
    edges = np.arange(-NB+1-0.5, NB-1+0.5+1, 1.0)  # [-14.5 .. 14.5]
    H, xedges, yedges = np.histogram2d(dx, dy, bins=[edges, edges])
    return H.astype(int), xedges, yedges


def _angular_maps(ix1, iy1, ix2, iy2, pitch_cm: float, distance_cm: float,
                  nbin_ang: int = 45):
    dx = ix2 - ix1
    dy = iy2 - iy1

    # ángulos observados
    thetax = np.degrees(np.arctan((dx * pitch_cm) / distance_cm))
    thetay = np.degrees(np.arctan((dy * pitch_cm) / distance_cm))
    rho = np.sqrt((dx*pitch_cm)**2 + (dy*pitch_cm)**2)
    theta = np.degrees(np.arctan(rho / distance_cm))

    # mapas θx,θy
    lim = np.degrees(np.arctan(((NB-1) * pitch_cm) / distance_cm))
    edges = np.linspace(-lim, lim, nbin_ang+1)
    Hxy, ex, ey = np.histogram2d(thetax, thetay, bins=[edges, edges])

    # hist θ (0..theta_max)
    tmax = np.degrees(np.arctan((math.sqrt(2)*(NB-1)*pitch_cm) / distance_cm))
    et = np.linspace(0.0, tmax, nbin_ang+1)
    Ht, et = np.histogram(theta, bins=et)

    return (Hxy.astype(int), ex, ey), (Ht.astype(int), et)


def _save_delta_csv(H, xedges, yedges, out_csv: str):
    # centros
    xc = 0.5*(xedges[:-1] + xedges[1:])
    yc = 0.5*(yedges[:-1] + yedges[1:])
    rows = []
    for i,dx in enumerate(xc):
        for j,dy in enumerate(yc):
            c = int(H[i,j])
            if c > 0:
                rows.append((int(round(dx)), int(round(dy)), c))
    pd.DataFrame(rows, columns=["dx_pix","dy_pix","counts"]).to_csv(out_csv, index=False)


def _save_ang_csv(Hxy, ex, ey, out_csv: str):
    xc = 0.5*(ex[:-1] + ex[1:])
    yc = 0.5*(ey[:-1] + ey[1:])
    rows=[]
    for i,tx in enumerate(xc):
        for j,ty in enumerate(yc):
            c=int(Hxy[i,j])
            if c>0:
                rows.append((tx,ty,c))
    pd.DataFrame(rows, columns=["theta_x_deg","theta_y_deg","counts"]).to_csv(out_csv, index=False)


def _save_theta_hist(Ht, et, out_csv: str):
    tc = 0.5*(et[:-1] + et[1:])
    pd.DataFrame({"theta_deg": tc, "counts": Ht}).to_csv(out_csv, index=False)


def _attenuation_test(csv_path: str, flip_y2: bool, mode: str, ch_offset: int,
                      covered_ch_start: int, covered_ch_end: int) -> dict:
    # covered range in channels -> local indices in Y2
    # Y2 channels are [ch_offset+45 .. ch_offset+59] (15 canales)
    y2_first = ch_offset + 45
    # map channel -> local index
    a = covered_ch_start - y2_first
    b = covered_ch_end - y2_first
    if not (0 <= a <= 14 and 0 <= b <= 14 and a <= b):
        return {"ok": False, "reason": f"Rango {covered_ch_start}-{covered_ch_end} no cae dentro de Y2 ({y2_first}-{y2_first+14})."}
    covered = set(range(a, b+1))

    # si hacemos flip_y2, el conjunto cubierto se refleja
    if flip_y2:
        covered = set((NB-1 - i) for i in covered)

    # hist Y2 con require4 (comparación consistente)
    _, _, _, iy2, keep_frac = _indices_for_events(csv_path, mode=mode, ch_offset=ch_offset,
                                                  flip_x2=False, flip_y2=flip_y2,
                                                  require4=True)
    hist = np.bincount(iy2, minlength=NB)
    cov = int(sum(hist[i] for i in covered))
    tot = int(hist.sum())
    unc = tot - cov

    R = cov/unc if unc>0 else float("nan")
    err = R*math.sqrt(1/max(cov,1) + 1/max(unc,1)) if cov>0 and unc>0 else float("nan")
    frac = cov/tot if tot>0 else float("nan")

    return {
        "ok": True,
        "keep_frac_4planes": keep_frac,
        "covered_local_indices": sorted(list(covered)),
        "counts_total": tot,
        "counts_cov": cov,
        "counts_unc": unc,
        "ratio_cov_over_unc": R,
        "ratio_err": err,
        "frac_cov": frac,
        "hist_y2": hist.tolist(),
    }


def _compare_two(lead_csv: str, ref_csv: str, flip_y2: bool, mode: str, ch_offset: int,
                 covered_ch_start: int, covered_ch_end: int) -> dict:
    A = _attenuation_test(lead_csv, flip_y2, mode, ch_offset, covered_ch_start, covered_ch_end)
    B = _attenuation_test(ref_csv, flip_y2, mode, ch_offset, covered_ch_start, covered_ch_end)
    if not (A.get("ok") and B.get("ok")):
        return {"ok": False, "lead": A, "ref": B}

    # diferencia de fracciones (binomial aprox)
    p1, n1 = A["frac_cov"], A["counts_total"]
    p2, n2 = B["frac_cov"], B["counts_total"]
    var = p1*(1-p1)/max(n1,1) + p2*(1-p2)/max(n2,1)
    z = (p1-p2)/math.sqrt(var) if var>0 else float("nan")

    return {"ok": True, "lead": A, "ref": B, "z_frac_diff": z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV principal (p.ej. con plomo).")
    ap.add_argument("--ref", default=None, help="CSV referencia (p.ej. sin plomo) para comparar atenuación.")
    ap.add_argument("--outdir", default=".", help="Directorio de salida.")
    ap.add_argument("--mode", choices=["centroid","strict"], default="centroid")
    ap.add_argument("--ch-offset", type=int, default=1, help="Primer canal útil (por defecto 1 -> ch01..ch60).")
    ap.add_argument("--pitch-cm", type=float, default=4.0)
    ap.add_argument("--distance-cm", type=float, default=30.0)
    ap.add_argument("--flip-x2", action="store_true", help="Invertir el orden de barras en X2.")
    ap.add_argument("--flip-y2", action="store_true", help="Invertir el orden de barras en Y2.")
    ap.add_argument("--nbin-ang", type=int, default=45)

    ap.add_argument("--lead-ch-start", type=int, default=46, help="Canal inicio del rango cubierto por plomo (inclusive).")
    ap.add_argument("--lead-ch-end", type=int, default=53, help="Canal fin del rango cubierto por plomo (inclusive).")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ix1, iy1, ix2, iy2, keep_frac = _indices_for_events(
        args.input, mode=args.mode, ch_offset=args.ch_offset,
        flip_x2=args.flip_x2, flip_y2=args.flip_y2, require4=True
    )

    H, xedges, yedges = _delta_maps(ix1, iy1, ix2, iy2)
    (Hxy, ex, ey), (Ht, et) = _angular_maps(ix1, iy1, ix2, iy2, args.pitch_cm, args.distance_cm, args.nbin_ang)

    out_delta = outdir / "mapa_pixeles_delta_xy.csv"
    out_ang = outdir / "mapa_angular_thetax_thetay.csv"
    out_theta = outdir / "hist_theta.csv"

    _save_delta_csv(H, xedges, yedges, str(out_delta))
    _save_ang_csv(Hxy, ex, ey, str(out_ang))
    _save_theta_hist(Ht, et, str(out_theta))

    print("=== Construcción de mapas ===")
    print(f"input: {args.input}")
    print(f"modo: {args.mode}")
    print(f"keep_frac_4planes: {keep_frac:.4f}")
    print(f"flip_x2={args.flip_x2}, flip_y2={args.flip_y2}")
    print(f"distance_cm={args.distance_cm}, pitch_cm={args.pitch_cm}")
    print(f"saved: {out_delta}")
    print(f"saved: {out_ang}")
    print(f"saved: {out_theta}")

    # test de atenuación en este archivo
    T = _attenuation_test(args.input, args.flip_y2, args.mode, args.ch_offset,
                          args.lead_ch_start, args.lead_ch_end)
    print("\n=== Test plomo (solo este archivo) ===")
    if not T["ok"]:
        print("No se pudo evaluar:", T.get("reason"))
    else:
        print(f"Rango plomo (canales): {args.lead_ch_start}-{args.lead_ch_end}")
        print(f"indices Y2 cubiertos (tras flip si aplica): {T['covered_local_indices']}")
        print(f"N_total={T['counts_total']}  N_cov={T['counts_cov']}  N_unc={T['counts_unc']}")
        print(f"frac_cov={T['frac_cov']:.6f}")
        print(f"R=cov/unc={T['ratio_cov_over_unc']:.4f} ± {T['ratio_err']:.4f}")

    # comparación con referencia
    if args.ref:
        C = _compare_two(args.input, args.ref, args.flip_y2, args.mode, args.ch_offset,
                         args.lead_ch_start, args.lead_ch_end)
        print("\n=== Comparación vs referencia ===")
        if not C["ok"]:
            print("Falló comparación.")
            print("lead:", C.get("lead"))
            print("ref :", C.get("ref"))
        else:
            print(f"ref: {args.ref}")
            print(f"frac_cov(lead)={C['lead']['frac_cov']:.6f}  frac_cov(ref)={C['ref']['frac_cov']:.6f}")
            print(f"z (diff de fracciones) = {C['z_frac_diff']:.3f}  (signo <0 sugiere atenuación en cubierto)")


if __name__ == "__main__":
    main()
