#!/usr/bin/env python3
"""
MuTe-like 2-plane correction workflow.

Inputs:
  data/mapa_pixeles_delta_xy.csv  (delta_x, delta_y, counts)
  data/bga_3600.csv               (ARTI sky; column theta)

Assumptions:
  - Two square planes L = Npix*pitch; Npix=15, pitch=4 cm, separation d=30 cm.
  - Overlap area A_ov(sx,sy) = (L-|sx|)(L-|sy|), clipped to >=0.
  - For bins uniform in (u,v) = (sx/d, sy/d): counts scale as
        N(u,v) ∝ I(θ) * A_ov(u,v) * cos^4(θ)

Outputs (figs/): (names match main.tex)
  mute_counts_map_raw.png
  mute_intensity_map_corrected.png
  mute_forward_pred_counts.png
  mute_forward_ratio_map.png
  mute_ratio_with_poisson_errors.png
  mute_recovered_intensity_vs_arti.png
  mute_epsilon_map.png
  mute_epsilon_vs_theta_fit.png
  mute_intensity_after_epsilon_correction.png
  mute_epsilon_map_thetaxy.png
  mute_gmap_anisotropy.png
  mute_g_slice_thetax.png
  mute_g_slice_thetay.png

Outputs (data/):
  mute_forward_diagnostics.csv
  mute_ratio_and_recovery_table.csv
  mute_epsilon_diagnostics.csv
  mute_anisotropy_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

# Geometry
a_cm, d_cm, Npix = 4.0, 30.0, 15
L_cm = Npix * a_cm
dmin, dmax = -(Npix - 1), (Npix - 1)
grid_size = 2 * (Npix - 1) + 1

# Load sparse map
det = pd.read_csv(DATA / "mapa_pixeles_delta_xy.csv")
dx = det["delta_x"].to_numpy()
dy = det["delta_y"].to_numpy()
cnt = det["counts"].to_numpy().astype(float)

N_obs = np.zeros((grid_size, grid_size), dtype=float)
for x, y, c in zip(dx, dy, cnt):
    if dmin <= x <= dmax and dmin <= y <= dmax:
        N_obs[y - dmin, x - dmin] = c

DX, DY = np.meshgrid(np.arange(dmin, dmax + 1), np.arange(dmin, dmax + 1))

# u,v and angles
u = (DX * a_cm) / d_cm
v = (DY * a_cm) / d_cm
cos_theta = 1.0 / np.sqrt(1.0 + u * u + v * v)
theta_deg = np.rad2deg(np.arccos(cos_theta))
theta_x = np.rad2deg(np.arctan(u))
theta_y = np.rad2deg(np.arctan(v))

# Overlap area
shift_x = np.abs(DX) * a_cm
shift_y = np.abs(DY) * a_cm
A_ov = (L_cm - shift_x) * (L_cm - shift_y)
A_ov[A_ov < 0] = 0.0

mask = (A_ov > 0) & (N_obs > 0)

# Raw map
plt.figure(figsize=(7.2, 6.2))
plt.imshow(N_obs, origin="lower",
           extent=[dmin - 0.5, dmax + 0.5, dmin - 0.5, dmax + 0.5],
           aspect="equal")
plt.xlabel(r"$\Delta x$ [pixel index]")
plt.ylabel(r"$\Delta y$ [pixel index]")
plt.title("Measured coincidence map (raw counts)")
plt.colorbar(label="Counts")
plt.tight_layout()
plt.savefig(FIGS / "mute_counts_map_raw.png", dpi=200)
plt.close()

# Geom-corrected intensity (up to scale)
I_est = np.full_like(N_obs, np.nan, dtype=float)
I_est[mask] = N_obs[mask] / (A_ov[mask] * (cos_theta[mask] ** 4 + 1e-15))
I_norm = I_est / np.nanmax(I_est)

plt.figure(figsize=(7.2, 6.2))
plt.imshow(I_norm, origin="lower",
           extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
           aspect="equal", vmin=0, vmax=1)
plt.xlabel(r"$\theta_x$ [deg]")
plt.ylabel(r"$\theta_y$ [deg]")
plt.title(r"Geom.-corrected map $I \propto N/(A_{\rm ov}\cos^4\theta)$ (normalized)")
plt.colorbar(label="Normalized intensity")
plt.tight_layout()
plt.savefig(FIGS / "mute_intensity_map_corrected.png", dpi=200)
plt.close()

# --- ARTI sky (same data used in "ARTI sky fit", but WITHOUT the fit) ---
arti_mu = pd.read_csv(DATA / "angular_binned_costheta_counts.csv")

mu_cent = arti_mu["mu_center"].to_numpy()
mu_counts = arti_mu["counts"].to_numpy().astype(float)

m = np.isfinite(mu_cent) & np.isfinite(mu_counts) & (mu_cent >= 0) & (mu_cent <= 1) & (mu_counts >= 0)
mu_cent = mu_cent[m]
mu_counts = mu_counts[m]

# normalize to compare shapes
I_mu = mu_counts / (mu_counts.max() if mu_counts.max() > 0 else 1.0)

def I_of_mu(mu):
    mu = np.clip(mu, 0.0, 1.0)
    return np.interp(mu, mu_cent, I_mu, left=I_mu[0], right=I_mu[-1])


# --- Forward prediction: ARTI×geom ---
shape = I_of_mu(cos_theta) * A_ov * (cos_theta ** 4)
x = shape[mask].ravel()
y = N_obs[mask].ravel()
C = (x @ y) / (x @ x)
N_pred = C * shape

plt.figure(figsize=(7.2, 6.2))
plt.imshow(N_pred, origin="lower",
           extent=[dmin - 0.5, dmax + 0.5, dmin - 0.5, dmax + 0.5],
           aspect="equal")
plt.xlabel(r"$\Delta x$ [pixel index]")
plt.ylabel(r"$\Delta y$ [pixel index]")
plt.title("Forward model prediction (ARTI sky × geometry)")
plt.colorbar(label="Predicted counts")
plt.tight_layout()
plt.savefig(FIGS / "mute_forward_pred_counts.png", dpi=200)
plt.close()

ratio_map = np.full_like(N_obs, np.nan, dtype=float)
ratio_map[mask] = N_obs[mask] / (N_pred[mask] + 1e-15)

plt.figure(figsize=(7.2, 6.2))
plt.imshow(ratio_map, origin="lower",
           extent=[dmin - 0.5, dmax + 0.5, dmin - 0.5, dmax + 0.5],
           aspect="equal", vmin=0.5, vmax=1.5)
plt.xlabel(r"$\Delta x$ [pixel index]")
plt.ylabel(r"$\Delta y$ [pixel index]")
plt.title("Ratio map: observed / predicted")
plt.colorbar(label="Obs/Pred")
plt.tight_layout()
plt.savefig(FIGS / "mute_forward_ratio_map.png", dpi=200)
plt.close()

# --- Radial bins in theta (sum estimator) ---
bin_w = 2.0
bins = np.arange(0, 80 + bin_w, bin_w)
th_cent = 0.5 * (bins[:-1] + bins[1:])
theta_flat = theta_deg[mask].ravel()
idx = np.digitize(theta_flat, bins) - 1

obs_sum = np.array([np.nansum(N_obs[mask][idx == i]) if np.any(idx == i) else np.nan
                    for i in range(len(th_cent))])
pred_sum = np.array([np.nansum(N_pred[mask][idx == i]) if np.any(idx == i) else np.nan
                     for i in range(len(th_cent))])

R = obs_sum / (pred_sum + 1e-15)
sigR = np.sqrt(obs_sum) / (pred_sum + 1e-15)

valid = np.isfinite(R) & np.isfinite(sigR) & (pred_sum > 0)
plt.figure(figsize=(8, 4.8))
plt.errorbar(th_cent[valid], R[valid], yerr=sigR[valid], fmt="o", capsize=2)
plt.axhline(1.0, lw=2)
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel("Obs/Pred")
plt.title("Obs/Pred vs θ with Poisson error bars")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "mute_ratio_with_poisson_errors.png", dpi=200)
plt.close()

pd.DataFrame({
    "theta_center_deg": th_cent,
    "obs_sum": obs_sum,
    "pred_sum": pred_sum,
    "ratio": R,
    "ratio_sigma_poisson": sigR
}).to_csv(DATA / "mute_forward_diagnostics.csv", index=False)

# --- Recover sky intensity from map (geom only) ---
den_geom = np.array([np.nansum((A_ov * (cos_theta ** 4))[mask][idx == i]) if np.any(idx == i) else np.nan
                    for i in range(len(th_cent))])
I_hat = obs_sum / (den_geom + 1e-15)
I_hat_n = I_hat / np.nanmax(I_hat)

I_arti_bin = I_of_mu(np.cos(np.deg2rad(th_cent)))
I_arti_n = I_arti_bin / np.nanmax(I_arti_bin)

fit = (th_cent <= 45) & (I_hat > 0)
n_hat, c_hat = np.polyfit(np.log(np.cos(np.deg2rad(th_cent[fit]))),
                          np.log(I_hat[fit]), 1)

th_plot = np.linspace(0, 60, 301)
model = np.exp(c_hat) * (np.cos(np.deg2rad(th_plot)) ** n_hat)
model = model / model.max()

plt.figure(figsize=(8, 4.8))
plt.plot(th_cent, I_hat_n, "o-", label="Recovered sky (geom only)")
plt.plot(th_cent, I_arti_n, "o-", label="ARTI sky (proxy)")
plt.plot(th_plot, model, lw=2, label=fr"Fit recovered: $n={n_hat:.2f}$ (θ≤45°)")
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel("Intensity (normalized)")
plt.title("Recovered sky vs ARTI")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "mute_recovered_intensity_vs_arti.png", dpi=200)
plt.close()

pd.DataFrame({
    "theta_center_deg": th_cent,
    "I_hat": I_hat,
    "I_hat_norm": I_hat_n,
    "I_arti_norm": I_arti_n
}).to_csv(DATA / "mute_ratio_and_recovery_table.csv", index=False)

# --- Epsilon map ---
eps_map = np.full_like(N_obs, np.nan, dtype=float)
eps_map[mask] = N_obs[mask] / (N_pred[mask] + 1e-15)

plt.figure(figsize=(7.2, 6.2))
plt.imshow(eps_map, origin="lower",
           extent=[dmin - 0.5, dmax + 0.5, dmin - 0.5, dmax + 0.5],
           aspect="equal", vmin=0.5, vmax=1.5)
plt.xlabel(r"$\Delta x$ [pixel index]")
plt.ylabel(r"$\Delta y$ [pixel index]")
plt.title(r"Residual factor $\varepsilon(\Delta x,\Delta y)$ = Obs/(ARTI×geom)")
plt.colorbar(label=r"$\varepsilon$")
plt.tight_layout()
plt.savefig(FIGS / "mute_epsilon_map.png", dpi=200)
plt.close()

# epsilon(theta) and fit eps ~ cos^k(theta)
eps_k = obs_sum / (pred_sum + 1e-15)
sig_eps = np.sqrt(obs_sum) / (pred_sum + 1e-15)

fit = (th_cent <= 45) & (eps_k > 0) & (sig_eps > 0)
mu = np.cos(np.deg2rad(th_cent[fit]))
x = np.log(mu)
y = np.log(eps_k[fit])
w = 1.0 / ((sig_eps[fit] / eps_k[fit]) ** 2)

S = np.sum(w); Sx = np.sum(w * x); Sy = np.sum(w * y)
Sxx = np.sum(w * x * x); Sxy = np.sum(w * x * y)
den = S * Sxx - Sx * Sx
k_fit = (S * Sxy - Sx * Sy) / den
b_fit = (Sy - k_fit * Sx) / S

th_plot = np.linspace(0, 60, 301)
eps_model = np.exp(b_fit) * (np.cos(np.deg2rad(th_plot)) ** k_fit)

plt.figure(figsize=(8, 4.8))
plt.errorbar(th_cent[valid], eps_k[valid], yerr=sig_eps[valid], fmt="o", capsize=2, label="ε per θ-bin")
plt.plot(th_plot, eps_model, lw=2, label=fr"Fit: ε∝cos^kθ, k={k_fit:.2f} (θ≤45°)")
plt.axhline(1.0, lw=1)
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel(r"$\varepsilon(\theta)$")
plt.title("Residual angular factor beyond geometry")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "mute_epsilon_vs_theta_fit.png", dpi=200)
plt.close()

# Apply epsilon radial correction to intensity
eps_fit_all = np.exp(b_fit) * (np.cos(np.deg2rad(th_cent)) ** k_fit)
I_clean = I_hat / (eps_fit_all + 1e-15)
I_clean_n = I_clean / np.nanmax(I_clean)

plt.figure(figsize=(8, 4.8))
plt.step(th_cent, I_hat_n, label="Recovered (geom only)")
plt.step(th_cent, I_clean_n, label="Recovered + ε-correction")
plt.step(th_cent, I_arti_n, label="ARTI sky")
plt.xlabel(r"Zenith angle $\theta$ [deg]")
plt.ylabel("Intensity (normalized)")
plt.title("Does ε-correction bring recovered sky closer to ARTI?")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "mute_intensity_after_epsilon_correction.png", dpi=200)
plt.close()

# anisotropy after removing eps_radial
eps_fit_2d = np.exp(b_fit) * (cos_theta ** k_fit)
g_map = np.full_like(eps_map, np.nan, dtype=float)
g_map[mask] = eps_map[mask] / (eps_fit_2d[mask] + 1e-15)

plt.figure(figsize=(7.2, 6.2))
plt.imshow(eps_map, origin="lower",
           extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
           aspect="equal", vmin=0.5, vmax=1.5)
plt.xlabel(r"$\theta_x$ [deg]")
plt.ylabel(r"$\theta_y$ [deg]")
plt.title(r"$\varepsilon(\theta_x,\theta_y)$")
plt.colorbar(label=r"$\varepsilon$")
plt.tight_layout()
plt.savefig(FIGS / "mute_epsilon_map_thetaxy.png", dpi=200)
plt.close()

plt.figure(figsize=(7.2, 6.2))
plt.imshow(g_map, origin="lower",
           extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
           aspect="equal", vmin=0.8, vmax=1.2)
plt.xlabel(r"$\theta_x$ [deg]")
plt.ylabel(r"$\theta_y$ [deg]")
plt.title(r"Residual anisotropy $g(\theta_x,\theta_y)$")
plt.colorbar(label="g (relative)")
plt.tight_layout()
plt.savefig(FIGS / "mute_gmap_anisotropy.png", dpi=200)
plt.close()

# simple slices
g = g_map[mask].ravel()
tx = theta_x[mask].ravel()
ty = theta_y[mask].ravel()
band = 5.0

gx = g[np.abs(ty) < band]; txb = tx[np.abs(ty) < band]
gy = g[np.abs(tx) < band]; tyb = ty[np.abs(tx) < band]

def binned(xv, yv, bw=2.0, xmin=-45, xmax=45):
    bins = np.arange(xmin, xmax + bw, bw)
    cen = 0.5 * (bins[:-1] + bins[1:])
    ii = np.digitize(xv, bins) - 1
    mean = np.array([np.mean(yv[ii == i]) if np.any(ii == i) else np.nan for i in range(len(cen))])
    sem = np.array([np.std(yv[ii == i], ddof=1) / np.sqrt(np.sum(ii == i)) if np.sum(ii == i) > 1 else np.nan
                    for i in range(len(cen))])
    return cen, mean, sem

cx, mx, sx = binned(txb, gx)
cy, my, sy = binned(tyb, gy)

plt.figure(figsize=(8, 4.8))
plt.errorbar(cx, mx, yerr=sx, fmt="o", capsize=2)
plt.axhline(1.0, lw=1)
plt.xlabel(r"$\theta_x$ [deg]")
plt.ylabel("g (relative)")
plt.title(r"Residual anisotropy slice: $g$ vs $\theta_x$ (|θy|<5°)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "mute_g_slice_thetax.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 4.8))
plt.errorbar(cy, my, yerr=sy, fmt="o", capsize=2)
plt.axhline(1.0, lw=1)
plt.xlabel(r"$\theta_y$ [deg]")
plt.ylabel("g (relative)")
plt.title(r"Residual anisotropy slice: $g$ vs $\theta_y$ (|θx|<5°)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "mute_g_slice_thetay.png", dpi=200)
plt.close()

# =========================
# FULL-CORRECTED ANGULAR MAP
# =========================

# epsilon radial in 2D using the fit: eps_rad(θ)=exp(b_fit)*cos^k_fit(θ)
eps_rad_2d = np.exp(b_fit) * (cos_theta ** k_fit)

# Full corrected intensity map:
# I_clean ∝ N_obs / (A_ov * cos^4θ * eps_rad(θ) * g(θx,θy))
I_clean_map = np.full_like(N_obs, np.nan, dtype=float)

den = (A_ov * (cos_theta ** 4) * eps_rad_2d * g_map)

good = mask & np.isfinite(den) & (den > 0)
I_clean_map[good] = N_obs[good] / den[good]

# Normalize for visualization (shape only)
I_clean_norm = I_clean_map / np.nanmax(I_clean_map)

plt.figure(figsize=(7.2, 6.2))
plt.imshow(I_clean_norm, origin="lower",
           extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
           aspect="equal", vmin=0, vmax=1)
plt.xlabel(r"$\theta_x$ [deg]")
plt.ylabel(r"$\theta_y$ [deg]")
plt.title("Fully corrected angular intensity (geom + ε_radial + g), normalized")
plt.colorbar(label="Normalized intensity")
plt.tight_layout()
plt.savefig(FIGS / "mute_intensity_map_fully_corrected.png", dpi=200)
plt.close()

# Optional: also save the unnormalized map for later quantitative use
pd.DataFrame({
    "theta_x_deg": theta_x[good].ravel(),
    "theta_y_deg": theta_y[good].ravel(),
    "theta_deg": theta_deg[good].ravel(),
    "I_clean": I_clean_map[good].ravel()
}).to_csv(DATA / "mute_intensity_map_fully_corrected.csv", index=False)

print("Saved figs/mute_intensity_map_fully_corrected.png")
print("Saved data/mute_intensity_map_fully_corrected.csv")

# ==========================================
# DUAL RESIDUAL MAPS (RADIAL-ONLY vs FULL)
# ==========================================

# --- ARTI sky in 2D (depends only on theta) ---
I_arti_2d = I_of_mu(np.cos(np.deg2rad(theta_deg)))  # already normalized shape in I_of_mu

eps_floor = 1e-15

# --- Geometry-only intensity (unnormalized shape) ---
I_geom_map = np.full_like(N_obs, np.nan, dtype=float)
den_geom = (A_ov * (cos_theta ** 4))
good_geom = mask & np.isfinite(den_geom) & (den_geom > 0)
I_geom_map[good_geom] = N_obs[good_geom] / (den_geom[good_geom] + eps_floor)

# --- epsilon radial in 2D from fit: eps_rad(θ)=exp(b_fit)*cos^k_fit(θ) ---
eps_rad_2d = np.exp(b_fit) * (cos_theta ** k_fit)

# --- Radial-only corrected intensity: geom + eps_rad ---
I_rad_map = np.full_like(N_obs, np.nan, dtype=float)
den_rad = den_geom * eps_rad_2d
good_rad = mask & np.isfinite(den_rad) & (den_rad > 0)
I_rad_map[good_rad] = N_obs[good_rad] / (den_rad[good_rad] + eps_floor)

# --- Full corrected intensity: geom + eps_rad + g ---
I_full_map = np.full_like(N_obs, np.nan, dtype=float)
den_full = den_rad * g_map
good_full = mask & np.isfinite(den_full) & (den_full > 0)
I_full_map[good_full] = N_obs[good_full] / (den_full[good_full] + eps_floor)

# --- Residual ratios vs ARTI ---
R_rad = np.full_like(N_obs, np.nan, dtype=float)
R_full = np.full_like(N_obs, np.nan, dtype=float)

goodR_rad  = good_rad  & np.isfinite(I_arti_2d) & (I_arti_2d > eps_floor) & np.isfinite(I_rad_map)
goodR_full = good_full & np.isfinite(I_arti_2d) & (I_arti_2d > eps_floor) & np.isfinite(I_full_map)

R_rad[goodR_rad]   = I_rad_map[goodR_rad] / (I_arti_2d[goodR_rad] + eps_floor)
R_full[goodR_full] = I_full_map[goodR_full] / (I_arti_2d[goodR_full] + eps_floor)

# Optional: log residuals (often nicer)
logR_rad  = np.full_like(N_obs, np.nan, dtype=float)
logR_full = np.full_like(N_obs, np.nan, dtype=float)
logR_rad[goodR_rad]   = np.log10(R_rad[goodR_rad])
logR_full[goodR_full] = np.log10(R_full[goodR_full])

# --- One figure, two panels (same color scale) ---
vmin, vmax = 0.5, 1.5   # ratio scale; adjust if needed

fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)

im0 = ax[0].imshow(R_rad, origin="lower",
                   extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
                   aspect="equal", vmin=vmin, vmax=vmax)
ax[0].set_title(r"Residual (geom + $\varepsilon_{\rm radial}$): $I_{\rm rad}/I_{\rm ARTI}(\theta)$")
ax[0].set_xlabel(r"$\theta_x$ [deg]")
ax[0].set_ylabel(r"$\theta_y$ [deg]")

im1 = ax[1].imshow(R_full, origin="lower",
                   extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
                   aspect="equal", vmin=vmin, vmax=vmax)
ax[1].set_title(r"Residual (geom + $\varepsilon_{\rm radial}$ + $g$): $I_{\rm full}/I_{\rm ARTI}(\theta)$")
ax[1].set_xlabel(r"$\theta_x$ [deg]")
ax[1].set_ylabel(r"$\theta_y$ [deg]")

cbar = fig.colorbar(im1, ax=ax, shrink=0.9)
cbar.set_label("Ratio")

plt.savefig(FIGS / "mute_residual_ratio_dualpanel_thetaxy.png", dpi=200)
plt.close()

# --- Same but log10 ratio (optional, very useful) ---
vminL, vmaxL = -0.3, 0.3
fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)

im0 = ax[0].imshow(logR_rad, origin="lower",
                   extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
                   aspect="equal", vmin=vminL, vmax=vmaxL)
ax[0].set_title(r"$\log_{10}(I_{\rm rad}/I_{\rm ARTI})$  (geom + $\varepsilon_{\rm radial}$)")
ax[0].set_xlabel(r"$\theta_x$ [deg]")
ax[0].set_ylabel(r"$\theta_y$ [deg]")

im1 = ax[1].imshow(logR_full, origin="lower",
                   extent=[theta_x.min(), theta_x.max(), theta_y.min(), theta_y.max()],
                   aspect="equal", vmin=vminL, vmax=vmaxL)
ax[1].set_title(r"$\log_{10}(I_{\rm full}/I_{\rm ARTI})$  (geom + $\varepsilon_{\rm radial}$ + $g$)")
ax[1].set_xlabel(r"$\theta_x$ [deg]")
ax[1].set_ylabel(r"$\theta_y$ [deg]")

cbar = fig.colorbar(im1, ax=ax, shrink=0.9)
cbar.set_label(r"$\log_{10}$ ratio")

plt.savefig(FIGS / "mute_residual_logratio_dualpanel_thetaxy.png", dpi=200)
plt.close()

# --- Save a compact CSV for later ---
good_any = np.isfinite(R_rad) | np.isfinite(R_full)
pd.DataFrame({
    "theta_x_deg": theta_x[good_any].ravel(),
    "theta_y_deg": theta_y[good_any].ravel(),
    "theta_deg": theta_deg[good_any].ravel(),
    "I_arti": I_arti_2d[good_any].ravel(),
    "I_rad": I_rad_map[good_any].ravel(),
    "I_full": I_full_map[good_any].ravel(),
    "R_rad": R_rad[good_any].ravel(),
    "R_full": R_full[good_any].ravel(),
    "logR_rad": logR_rad[good_any].ravel(),
    "logR_full": logR_full[good_any].ravel(),
}).to_csv(DATA / "mute_residual_dualpanel_maps_thetaxy.csv", index=False)

print("Saved:")
print(" - figs/mute_residual_ratio_dualpanel_thetaxy.png")
print(" - figs/mute_residual_logratio_dualpanel_thetaxy.png")
print(" - data/mute_residual_dualpanel_maps_thetaxy.csv")

def mean_sem(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(arr.size))

mL, seL = mean_sem(g[tx < 0]); mR, seR = mean_sem(g[tx > 0])
A_LR = (mR - mL) / (0.5 * (mR + mL))
mD, seD = mean_sem(g[ty < 0]); mU, seU = mean_sem(g[ty > 0])
A_UD = (mU - mD) / (0.5 * (mU + mD))

pd.DataFrame({
    "metric": ["k_fit", "A_LR", "A_UD",
               "mean_g_tx<0", "sem_g_tx<0", "mean_g_tx>0", "sem_g_tx>0",
               "mean_g_ty<0", "sem_g_ty<0", "mean_g_ty>0", "sem_g_ty>0"],
    "value": [k_fit, A_LR, A_UD, mL, seL, mR, seR, mD, seD, mU, seU]
}).to_csv(DATA / "mute_anisotropy_summary.csv", index=False)

pd.DataFrame({
    "theta_center_deg": th_cent,
    "epsilon": eps_k,
    "epsilon_sigma_poisson": sig_eps,
    "epsilon_fit": eps_fit_all,
    "I_hat": I_hat,
    "I_clean": I_clean
}).to_csv(DATA / "mute_epsilon_diagnostics.csv", index=False)

print("Done. k_fit =", k_fit, "A_LR =", A_LR, "A_UD =", A_UD)
