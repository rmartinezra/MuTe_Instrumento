#!/usr/bin/env python3
"""01_arti_histograms_v2.py

From an ARTI event list (one row per muon at ground), build *absolute* zenith-angle
intensity curves assuming the file represents a physical exposure:

  N_events(θ-bin) = I(θ) * A_ref * Δt * ΔΩ_bin

where I(θ) = dN/(dA_perp dΩ dt) has units [m^-2 sr^-1 s^-1] (or [cm^-2 sr^-1 s^-1]).

Key assumption (must match how ARTI was configured):
- The CSV contains all muons crossing a reference horizontal plane of area A_ref
  during an exposure Δt (default: A_ref = 1 m^2, Δt = 3600 s).

Inputs:
  data/bga_3600.csv   (column: theta in degrees)

Outputs (data/):
  arti_Itheta_from_theta_bins.csv     (I(θ) using θ bins, azimuth integrated)
  arti_Itheta_from_mu_bins.csv        (I(θ) using uniform μ=cosθ bins)
  arti_meta.json                      (assumptions used)

Outputs (figs/):
  arti_Itheta_theta_bins.png
  arti_Itheta_mu_bins.png
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def delta_omega_theta_bin(theta_lo_deg: float, theta_hi_deg: float) -> float:
    """Azimuth-integrated solid angle for a zenith bin [θ_lo, θ_hi]."""
    lo = np.deg2rad(theta_lo_deg)
    hi = np.deg2rad(theta_hi_deg)
    return 2.0 * np.pi * (np.cos(lo) - np.cos(hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/bga_3600.csv", help="ARTI CSV with a 'theta' column (deg).")
    ap.add_argument("--dt", type=float, default=3600.0, help="Exposure time represented by the ARTI file [s].")
    ap.add_argument("--area", type=float, default=1.0, help="Reference horizontal area represented by the ARTI file [m^2].")
    ap.add_argument("--theta-bin", type=float, default=1.0, help="Bin width in θ [deg] for θ-binned curve.")
    ap.add_argument("--mu-bins", type=int, default=50, help="Number of uniform bins in μ=cosθ for μ-binned curve.")
    ap.add_argument("--outdir", default=".", help="Project root containing data/ and figs/.")
    args = ap.parse_args()

    ROOT = Path(args.outdir).resolve()
    DATA = ROOT / args.input
    OUTD = ROOT / "data"
    FIGS = ROOT / "figs"
    OUTD.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA)
    theta = df["theta"].to_numpy()
    theta = theta[np.isfinite(theta)]
    theta = theta[(theta >= 0) & (theta <= 90)]

    # ---- 1) I(θ) from uniform θ bins (azimuth integrated)
    bw = float(args.theta_bin)
    edges = np.arange(0, 90 + bw, bw)
    counts, edges = np.histogram(theta, bins=edges)
    cent = 0.5 * (edges[:-1] + edges[1:])

    dOmega = np.array([delta_omega_theta_bin(edges[i], edges[i+1]) for i in range(len(edges)-1)])
    # Intensity per unit perpendicular area:
    # N = I * (A_ref) * dt * dΩ  (BUT A_ref is horizontal area; perpendicular area = A_ref*cosθ)
    # For a horizontal reference plane, the number crossing is N = I(θ) * (A_ref*cosθ) * dt * dΩ.
    # => I(θ) = N / (A_ref * cosθ * dt * dΩ)
    cos_cent = np.cos(np.deg2rad(cent))
    I_theta = np.full_like(cent, np.nan, dtype=float)
    m = (dOmega > 0) & (cos_cent > 1e-6)
    I_theta[m] = counts[m] / (args.area * cos_cent[m] * args.dt * dOmega[m])  # [m^-2 sr^-1 s^-1]

    out1 = pd.DataFrame({
        "theta_center_deg": cent,
        "counts": counts,
        "dOmega_sr": dOmega,
        "cos_theta": cos_cent,
        "I_m2_sr_s": I_theta,
        "I_cm2_sr_s": I_theta / 1e4,
    })
    out1.to_csv(OUTD / "arti_Itheta_from_theta_bins.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(cent[m], I_theta[m])
    plt.yscale("log")
    plt.xlabel(r"Zenith angle $\theta$ (deg)")
    plt.ylabel(r"$I(\theta)$  [m$^{-2}$ sr$^{-1}$ s$^{-1}$]")
    plt.title("ARTI-derived intensity from θ bins (assumes A_ref, Δt)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "arti_Itheta_theta_bins.png", dpi=200)
    plt.close()

    # ---- 2) I(θ) from uniform μ bins (μ=cosθ)
    mu = np.cos(np.deg2rad(theta))
    mu_edges = np.linspace(0, 1, int(args.mu_bins) + 1)
    mu_counts, mu_edges = np.histogram(mu, bins=mu_edges)
    mu_cent = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    # For azimuth-integrated rings, ΔΩ = 2π Δμ
    dOmega_mu = 2.0 * np.pi * (mu_edges[1:] - mu_edges[:-1])

    theta_cent = np.rad2deg(np.arccos(np.clip(mu_cent, 0, 1)))
    cos_cent2 = mu_cent
    I_mu = np.full_like(mu_cent, np.nan, dtype=float)
    m2 = (dOmega_mu > 0) & (cos_cent2 > 1e-6)
    I_mu[m2] = mu_counts[m2] / (args.area * cos_cent2[m2] * args.dt * dOmega_mu[m2])

    out2 = pd.DataFrame({
        "mu_center": mu_cent,
        "theta_center_deg": theta_cent,
        "counts": mu_counts,
        "dOmega_sr": dOmega_mu,
        "cos_theta": cos_cent2,
        "I_m2_sr_s": I_mu,
        "I_cm2_sr_s": I_mu / 1e4,
    })
    out2.to_csv(OUTD / "arti_Itheta_from_mu_bins.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(theta_cent[m2], I_mu[m2])
    plt.yscale("log")
    plt.xlabel(r"Zenith angle $\theta$ (deg)")
    plt.ylabel(r"$I(\theta)$  [m$^{-2}$ sr$^{-1}$ s$^{-1}$]")
    plt.title("ARTI-derived intensity from uniform μ bins (assumes A_ref, Δt)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "arti_Itheta_mu_bins.png", dpi=200)
    plt.close()

    meta = {
        "assumption": "ARTI CSV rows are muons crossing a horizontal reference plane of area A_ref during Δt.",
        "A_ref_m2": args.area,
        "dt_s": args.dt,
        "theta_bin_deg": bw,
        "mu_bins": int(args.mu_bins),
        "note": "If ARTI output is normalized differently, update A_ref and/or dt accordingly."
    }
    (OUTD / "arti_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
