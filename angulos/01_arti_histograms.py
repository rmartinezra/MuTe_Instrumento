#!/usr/bin/env python3
"""
ARTI angular histograms and conversions:
- Histogram in theta (uniform bins)
- Sine-corrected curve ~ dI/dOmega
- Histogram in cos(theta)
- Fit ARTI sky to cos^n(theta) using bins uniform in cos(theta)

Inputs:
  data/bga_3600.csv   (column: theta)

Outputs (figs/):
  angular_hist_theta_counts.png
  angular_hist_theta_sine_corrected.png
  angular_hist_costheta_counts.png
  arti_sky_fit_costheta.png

Output (data/):
  angular_binned_theta_sine_corrected.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "data" / "bga_3600.csv"
FIGS = ROOT / "figs"
OUTD = ROOT / "data"
FIGS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
theta = df["theta"].to_numpy()
theta = theta[np.isfinite(theta)]
theta = theta[(theta >= 0) & (theta <= 90)]

# Histogram in theta
bin_w = 1.0
bins = np.arange(0, 90 + bin_w, bin_w)
counts, edges = np.histogram(theta, bins=bins)
cent = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(8, 4.8))
plt.step(cent, counts, where="mid")
plt.xlabel(r"Zenith angle $\theta$ (deg)")
plt.ylabel("Counts per 1° bin")
plt.title("Angular histogram (ARTI output)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "angular_hist_theta_counts.png", dpi=200)
plt.close()

# Sine correction
sin_cent = np.sin(np.deg2rad(cent))
mask = sin_cent > 1e-6
corr = np.full_like(counts, np.nan, dtype=float)
corr[mask] = counts[mask] / sin_cent[mask]
corr_norm = corr / np.nanmax(corr)

plt.figure(figsize=(8, 4.8))
plt.plot(cent[mask], corr_norm[mask])
plt.xlabel(r"Zenith angle $\theta$ (deg)")
plt.ylabel(r"$(dN/d\theta)/\sin\theta$ (normalized)")
plt.title(r"Sine-corrected angular distribution ~ $dI/d\Omega$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "angular_hist_theta_sine_corrected.png", dpi=200)
plt.close()

pd.DataFrame({
    "theta_center_deg": cent,
    "counts_per_1deg": counts,
    "sin_theta": sin_cent,
    "counts_over_sin": corr
}).to_csv(OUTD / "angular_binned_theta_sine_corrected.csv", index=False)

# Histogram in cos(theta)
mu = np.cos(np.deg2rad(theta))
mu_bins = np.linspace(0, 1, 51)  # 50 bins
mu_counts, mu_edges = np.histogram(mu, bins=mu_bins)
mu_cent = 0.5 * (mu_edges[:-1] + mu_edges[1:])
# Save cos(theta) binned distribution (the one used for ARTI sky fit, WITHOUT the fit)
pd.DataFrame({
    "mu_center": mu_cent,
    "counts": mu_counts
}).to_csv(OUTD / "angular_binned_costheta_counts.csv", index=False)



plt.figure(figsize=(8, 4.8))
plt.step(mu_cent, mu_counts, where="mid")
plt.xlabel(r"$\cos\theta$")
plt.ylabel("Counts per uniform bin")
plt.title("Histogram in cos(theta) (uniform bins)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "angular_hist_costheta_counts.png", dpi=200)
plt.close()

# Fit to cos^n(theta) using uniform mu bins, restrict to theta<=40 deg
theta_cent = np.rad2deg(np.arccos(mu_cent))
mfit = (mu_counts > 0) & (mu_cent > 0) & (theta_cent <= 80)
x = np.log(mu_cent[mfit])
y = np.log(mu_counts[mfit])
n_fit, c_fit = np.polyfit(x, y, 1)

th_plot = np.linspace(0, 80, 301)
model = np.exp(c_fit) * (np.cos(np.deg2rad(th_plot)) ** n_fit)
model = model / model.max()

plt.figure(figsize=(8, 4.8))
plt.step(theta_cent, mu_counts / mu_counts.max())
plt.plot(th_plot, model, lw=2)
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel("ARTI sky distribution (proxy, normalized)")
plt.title(fr"ARTI sky fit: $\propto \cos^n\theta$, $n={n_fit:.2f}$ (θ≤80°)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "arti_sky_fit_costheta.png", dpi=200)
plt.close()

print("Done. n_fit =", n_fit)
