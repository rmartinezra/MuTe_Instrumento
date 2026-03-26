#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone muography mapper:
- Reconstructs raw angular counts from an event CSV
- Applies pixel-weight and geometric correction
- Generates 3 GIFs: raw, corrected, error
- Generates 3 final PNG maps with article-style aesthetics

Usage
-----
python muography_unified_article_gif.py archivo.csv

Notes
-----
- No command-line flags are required. Edit the CONFIG block below if you want to
  change geometry, time binning, color scales, titles, etc.
- If the input is a delta-map (delta_x, delta_y, counts), the script will still
  generate the final PNG maps, but GIFs are skipped because there is no time axis.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps
from PIL import Image

# =============================================================================
# USER CONFIGURATION
# Edit this block first. The rest of the script reads these values.
# =============================================================================
CONFIG = {
    # Output folder. If None, a folder based on the input name is created.
    "output_dir": None,

    # Geometry and reconstruction defaults
    "pitch_cm": 4.0,
    "length_cm": 60.0,
    "distance_cm": 200.0,
    "tilt_y_deg": 12.0,
    "theta_max_deg": 26.0,
    "flip_y1": True,
    "flip_y2": True,

    # Time handling
    "step_s": 10800.0,                 # GIF frame width in seconds
    "gif_fps": 4,
    "gif_mode": "cumulative",         # "cumulative" or "incremental"
    "live_threshold_s": 120.0,        # dt above this is not counted as live time

    # Error map
    "error_kind": "relative",         # "relative" [%] or "absolute"

    # Figure styling
    "figure_dpi": 180,
    "png_dpi": 320,
    "gif_frame_dpi": 180,
    "font_family": "DejaVu Sans",
    "base_fontsize": 12,
    "title_fontsize": 13,
    "label_fontsize": 13,
    "tick_fontsize": 11,
    "annot_fontsize": 10,
    "linewidth_axes": 1.2,
    "tick_length": 5.5,
    "tick_width": 1.1,
    "figure_facecolor": "white",
    "axes_facecolor": "white",
    "ylim_global_deg": (-4.0, 26.0),

    # Colormaps: built-in, perceptually uniform Matplotlib maps
    "colormaps": {
        "raw": "viridis",
        "corrected": "cividis",
        "error": "magma",
    },

    # Color scale control. Edit here to change scales quickly.
    # mode = "auto" uses the final full map.
    # mode = "fixed" uses the vmin/vmax below.
    "color_scales": {
        "raw": {
            "mode": "auto",
            "vmin": 1.0,
            "vmax": None,
            "robust_percentiles": (2.0, 98.0),
        },
        "corrected": {
            "mode": "auto",
            "vmin": 1.0,
            "vmax": None,
            "robust_percentiles": (2.0, 98.0),
        },
        "error": {
            "mode": "auto",
            "vmin": 0.0,
            "vmax": None,
            "robust_percentiles": (2.0, 98.0),
        },
    },

    # Save all frame PNGs in addition to the GIFs
    "keep_frame_pngs": False,
}

# =============================================================================
# Embedded pixel-weight correction matrices from the user's base script
# =============================================================================
W1 = np.array([[1.1450349583229407, 1.1844346240890193, 0.9956621683948031, 1.2939723866920956, 1.3460712892020232,
  1.5809183605708184, 2.06591271057509  , 1.4752602080282242, 1.5144534583518285, 1.4121410595507466,
  1.070145306650085 , 1.166912634589903 , 1.5741343783667472, 1.2760240140971424, 2.3350192404594003],
 [0.7995133614366141, 0.8230437028872553, 0.8171594309518044, 0.8540250595281512, 0.8381460852649967,
  0.873464872322301 , 1.2702873106157693, 0.8795096668766559, 0.854928617501345 , 0.7927523643870833,
  0.7178689476900423, 0.6840963289168219, 0.7568876260646343, 0.7342190608524602, 1.5245268401475465],
 [0.8396622795623984, 0.8796068595159784, 0.8758216956261091, 0.8791245038243491, 0.8348090424735649,
  0.8465535297625552, 1.1548599341657557, 0.770295285758216 , 0.7525570991730623, 0.692655781483531 ,
  0.6411870396179385, 0.5986516140213334, 0.5917770875779615, 0.5593101525615741, 1.3749034520797152],
 [0.9504788280631121, 0.9893674877581751, 0.9233245258436491, 0.9691932078950285, 0.9719557140096776,
  1.0082558222291929, 1.3303893617475586, 0.9383356802546526, 0.8820435440720351, 0.8156801025911936,
  0.7232532148103333, 0.6911573192771935, 0.7465145285928099, 0.6583431953758496, 1.3945819107738417],
 [0.9943513232352091, 1.0648990375471237, 1.006062748518477 , 1.0668365082029303, 1.091995751608973 ,
  1.223767532606614 , 1.5832125874938154, 1.2155655571653181, 1.2052833585961311, 1.0202663589196905,
  0.8965884312227594, 0.8541941465411815, 1.2138873020997343, 0.8721739052997709, 1.6807128003481873],
 [0.9811428202530902, 1.0327335052048983, 1.0281224912382694, 1.044588219999407 , 1.0328332108080884,
  1.0723329909087467, 1.506044011574809 , 1.0524351137569699, 1.0211463939195429, 0.90294233695746  ,
  0.8448784375127496, 0.7991211471059978, 0.9369788774982725, 0.8026146403273576, 1.7851321498684962],
 [1.0556097870471564, 1.1669147729762017, 1.099177030826065 , 1.2252772675778894, 1.2714033855789049,
  1.4768935500936158, 1.7663961517638527, 1.4567383218986893, 1.3419211545489127, 1.0834546048203377,
  0.9792396829889559, 0.9018825677057716, 1.3374293767836645, 0.9098384896663394, 1.6685604118173973],
 [1.010977263714445 , 1.1268359318090346, 1.072419607800125 , 1.1143239447151554, 1.1531174648349476,
  1.2586306470114974, 1.629757012831577 , 1.2018896003278805, 1.1733191251958974, 1.0133985352098638,
  0.9309162060194799, 0.8597221542796384, 1.1460226103270166, 0.826556469888717 , 1.7690494956849503],
 [0.9957458520707402, 1.139249274985234 , 1.0610473946904386, 1.1822368285276157, 1.2354262543560748,
  1.520821776792535 , 1.7335784957977454, 1.35509430286499  , 1.2718734782551615, 1.0493562257352773,
  0.9518844892372105, 0.8986192253478891, 1.3190934029576822, 0.8646787834637855, 1.7992579549504237],
 [0.9604027753975889, 1.0360682998791133, 1.031850021304804 , 1.0477826689402152, 1.0235085889093953,
  1.095917541986628 , 1.440687871130033 , 1.0448851210810641, 0.9902499868008128, 0.9484437610156846,
  0.8431190008692759, 0.7994433004480129, 0.9151698356629439, 0.7366772100444993, 1.4665641201116457],
 [0.9513903944606024, 1.078816510673179 , 1.006835135259539 , 1.0835325510443412, 1.0790538413654611,
  1.24466429917579  , 1.5624120745654044, 1.118902527038485 , 0.9996614554436352, 0.9592892585210147,
  0.8662751230309708, 0.8337339250502958, 1.0142718109076732, 0.7380476643914855, 1.4895997251467152],
 [0.8595460402181198, 0.9149628240457536, 0.939784865440522 , 0.9474253830506832, 0.9554060300017434,
  0.9742364595395797, 1.3627146272883799, 0.9246876415556562, 0.8942985412724254, 0.8650485730493436,
  0.7758415953615054, 0.7418816864716608, 0.8018628082443956, 0.6570628433247518, 1.2895656619848448],
 [0.8097359391645024, 0.940613723039823 , 0.9050145163089003, 0.937101171389914 , 0.9712499013982795,
  1.123166682643213 , 1.3591189799676586, 1.0052325085378864, 0.8808511807561561, 0.8740024085797274,
  0.7863073683771029, 0.7566501249307435, 0.9181789959380199, 0.6874156810442186, 1.5411318911836438],
 [0.7329896347321169, 0.7949612207251002, 0.816224916966596 , 0.8581098810872821, 0.8660234149397302,
  0.9098412664253415, 1.2252042287839884, 0.8810612323497627, 0.8238918638655068, 0.7936081070901194,
  0.7375295917176958, 0.6975070241593657, 0.7832836411457469, 0.5912123759953848, 1.451374896956122 ],
 [1.062797877392664 , 1.2701777042023465, 1.1468291217814888, 1.1822809374708743, 1.2067762812267449,
  1.6647324502047467, 1.7932249805991187, 1.3431523879524627, 1.1706885165211813, 1.1625569690690256,
  1.0571511706111831, 1.0163325478780039, 1.4321528257719376, 0.914786084103603 , 2.932865420199934 ]])

W2 = np.array([[3.9678151662466346, 2.7629010466980057, 1.1399936559971164, 2.010538447621972 , 1.6334881446562448,
  1.744005618631944 , 3.9162953347114047, 2.7498079239259616, 4.425569466661159 , 0.8050546476418213,
  5.150777264191862 , 5.726120527111402 , 0.7455156104437536, 2.4933618874520653, 3.772394240576687 ],
 [0.839502661286806 , 0.5887982288370969, 0.5574178861129503, 0.6409192581232485, 0.598328721123853 ,
  0.6342974789229813, 0.7964784133155545, 0.78083905702173  , 0.8475613785155778, 0.4717631738022437,
  0.9094696759065525, 0.8892216973398585, 0.4304147994776789, 0.7644879658838579, 1.0698720734463043],
 [0.6874710126913314, 0.5348945641089436, 0.5374937867105265, 0.5516304001361637, 0.5667627734140926,
  0.577958153544138 , 0.5994159242470821, 0.5961659633339059, 0.626767000535299 , 0.5039315880818557,
  0.5964217786026871, 0.5764125183328154, 0.4179624873801356, 0.5078039693085734, 0.7796395156910882],
 [2.6501752571472545, 1.5142078847409894, 1.0069250542134227, 1.2582266498290922, 0.976582151179033 ,
  1.0402098285653523, 1.7527629974881407, 1.4730564128644976, 1.986756278134395 , 0.6621443902270856,
  2.2768939215838175, 2.1137463132659438, 0.5834501890331523, 1.0800650555271973, 1.5918105147028208],
 [0.9307504383508133, 0.7231125460317487, 0.7075625449229728, 0.7282154190363556, 0.7338091992093146,
  0.7281286717059626, 0.7852007322548479, 0.7709608935497425, 0.8018612565944513, 0.6612409486414843,
  0.7791048517910388, 0.7690862108256589, 0.5769669451627544, 0.7373387089913569, 1.2114217397731404],
 [1.9785765899499401, 1.4966051067712822, 1.1742798648525556, 1.492772894664748 , 1.1886946536072098,
  1.1054166585225018, 2.057113104856663 , 1.6606765861604882, 2.340718124230088 , 0.8185980626084884,
  2.308230522148753 , 2.05060460211249  , 0.7027615336412393, 1.0831666185274265, 1.4203364859625724],
 [0.9619122661037428, 0.7873387438178749, 0.8041882444205056, 0.8312962430166538, 0.8662573149854194,
  0.8510689004795169, 0.8997713645247436, 0.8944972066659596, 0.9233065781188762, 0.7854551698817686,
  0.8930742955064699, 0.8461254209592419, 0.7032738487741954, 0.7853319577947692, 1.1155271737064234],
 [1.6315113004252026, 1.2364923348219836, 1.0455105571511303, 1.1366981333170667, 1.1090543547073366,
  1.0121365663787731, 1.3575421445806948, 1.2983665424213187, 1.4947308057059343, 0.9240033659859859,
  1.4428313638384942, 1.3665308247179708, 0.7947286741509128, 1.0371222040641437, 1.343951310112326 ],
 [1.4847140114310289, 1.1193528016706384, 0.9605216628839731, 1.0549715124794161, 1.0761448344508466,
  0.9865686402000263, 1.4287413310377517, 1.4645792558320472, 2.263426112440749 , 0.9413805092396461,
  2.0193757603313425, 1.9173571136913066, 0.8546688488036295, 1.1377100328434697, 1.8502762327058664],
 [2.797540443200218 , 2.0190380437380377, 1.2646796963184717, 1.644731108277895 , 1.291791856570965 ,
  1.0434266242309194, 2.4216548230397414, 1.897038853571777 , 2.8951976075396684, 0.9447255200432496,
  1.8744360474725341, 1.712713608235764 , 0.8437389704599435, 1.1128621418340807, 1.3347653494642502],
 [1.225946775254125 , 1.0094346279746085, 0.9573806712037193, 1.039024131524449 , 1.036617067528288 ,
  0.9731910741245305, 1.2582380330379879, 1.2288629698335323, 1.4430807976366802, 0.9345112756840301,
  1.3975691542730337, 1.420653426603549 , 0.8596576623556373, 1.0365853577013309, 1.2898949889351436],
 [0.999052130729773 , 0.8502547422613939, 0.8758754321526854, 0.9371298629471213, 0.9643742513991902,
  0.944996477563649 , 1.1596134567487635, 1.118346339742221 , 1.3000833244938912, 0.916059436836668 ,
  1.232550391263298 , 1.261137120898369 , 0.8225524943603002, 0.9278763170638945, 1.1512351811889583],
 [1.5948807240208687, 1.7220713596653456, 1.0065480495295631, 1.311149950587798 , 1.0379704613869327,
  0.9593770835631054, 1.9599517457975848, 1.5044587862656673, 2.475574279795107 , 0.8938263800878327,
  1.688611568850179 , 1.6804747448110162, 0.8110161305224053, 0.9468805453093954, 1.1813584273260207],
 [0.8709913226386897, 0.7860676783456425, 0.7400126674026949, 0.836112066405934 , 0.8297124762337418,
  0.8202667674592096, 1.02013755181082  , 0.9909077577357527, 1.1072854638382412, 0.7974670073457375,
  1.0419722048249314, 1.0183189179895098, 0.7289756534934192, 0.8025768244419882, 0.9781748382253589],
 [0.9664622529446382, 0.8444630089328312, 0.7904795216081563, 1.0148411736914411, 0.8419364744566634,
  0.8048783223919228, 1.357193779511215 , 1.1141398343318165, 1.5260277757368697, 0.7454053307555188,
  1.405639420132771 , 1.3359972363134704, 0.724687449678571 , 0.8563579768917398, 1.0936028459272713]])


# Channel convention kept from the user's scripts:
# panel 1: X1 ch01-15, Y1 ch16-30
# panel 2: X2 ch32-46, Y2 ch47-61
X1_COLS = [f"ch{i:02d}" for i in range(1, 16)]
Y1_COLS = [f"ch{i:02d}" for i in range(16, 31)]
X2_COLS = [f"ch{i:02d}" for i in range(32, 47)]
Y2_COLS = [f"ch{i:02d}" for i in range(47, 62)]
ALL_EVENT_COLS = ["time"] + X1_COLS + Y1_COLS + X2_COLS + Y2_COLS


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": CONFIG["font_family"],
        "font.size": CONFIG["base_fontsize"],
        "axes.labelsize": CONFIG["label_fontsize"],
        "axes.titlesize": CONFIG["title_fontsize"],
        "xtick.labelsize": CONFIG["tick_fontsize"],
        "ytick.labelsize": CONFIG["tick_fontsize"],
        "xtick.major.size": CONFIG["tick_length"],
        "ytick.major.size": CONFIG["tick_length"],
        "xtick.major.width": CONFIG["tick_width"],
        "ytick.major.width": CONFIG["tick_width"],
        "axes.linewidth": CONFIG["linewidth_axes"],
        "figure.dpi": CONFIG["figure_dpi"],
        "savefig.dpi": CONFIG["png_dpi"],
        "figure.facecolor": CONFIG["figure_facecolor"],
        "axes.facecolor": CONFIG["axes_facecolor"],
    })


def detect_input_type(path: Path) -> str:
    cols = list(pd.read_csv(path, nrows=0).columns)
    cols_lower = {c.lower() for c in cols}
    if {"delta_x", "delta_y"}.issubset(cols_lower):
        return "map"
    if all(c in cols for c in ALL_EVENT_COLS):
        return "events"
    raise ValueError(
        "Input not recognized. Use either an event CSV with time,ch01..ch61 "
        "or a delta-map CSV with delta_x, delta_y, counts."
    )


def plane_centroid(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sums = arr.sum(axis=1)
    idx = np.arange(15, dtype=float)
    cent = np.divide(arr @ idx, sums, out=np.full(arr.shape[0], np.nan), where=sums > 0)
    return cent, sums


def rect_solid_angle(x0: float, x1: float, y0: float, y1: float, z: float) -> float:
    def f(x: float, y: float) -> float:
        return np.arctan2(x * y, z * np.sqrt(z * z + x * x + y * y))
    return float(f(x1, y1) - f(x0, y1) - f(x1, y0) + f(x0, y0))


def reconstruct_event_table(path: Path, chunksize: int = 200_000) -> tuple[pd.DataFrame, dict]:
    rows = []
    total_rows = 0
    valid_events = 0
    multi_hit_events = 0

    for chunk in pd.read_csv(path, usecols=ALL_EVENT_COLS, chunksize=chunksize):
        total_rows += len(chunk)
        t = pd.to_datetime(chunk["time"], errors="coerce")

        X1 = chunk[X1_COLS].to_numpy(float)
        Y1 = chunk[Y1_COLS].to_numpy(float)
        X2 = chunk[X2_COLS].to_numpy(float)
        Y2 = chunk[Y2_COLS].to_numpy(float)

        cX1, sX1 = plane_centroid(X1)
        cY1, sY1 = plane_centroid(Y1)
        cX2, sX2 = plane_centroid(X2)
        cY2, sY2 = plane_centroid(Y2)

        valid = (sX1 > 0) & (sY1 > 0) & (sX2 > 0) & (sY2 > 0) & t.notna().to_numpy()
        if not np.any(valid):
            continue

        valid_events += int(valid.sum())
        multi_hit_events += int((((sX1 > 1) | (sY1 > 1) | (sX2 > 1) | (sY2 > 1)) & valid).sum())

        iX1_raw = np.rint(cX1[valid]).astype(int)
        iY1_raw = np.rint(cY1[valid]).astype(int)
        iX2_raw = np.rint(cX2[valid]).astype(int)
        iY2_raw = np.rint(cY2[valid]).astype(int)

        iX1 = iX1_raw.copy()
        iX2 = iX2_raw.copy()
        iY1 = 14 - iY1_raw if CONFIG["flip_y1"] else iY1_raw.copy()
        iY2 = 14 - iY2_raw if CONFIG["flip_y2"] else iY2_raw.copy()

        dx = iX2 - iX1
        dy = iY2 - iY1

        u_cm = dx.astype(float) * CONFIG["pitch_cm"]
        v_cm = dy.astype(float) * CONFIG["pitch_cm"]
        theta_deg = np.degrees(np.arctan2(np.sqrt(u_cm**2 + v_cm**2), CONFIG["distance_cm"]))

        overlap_m2 = (
            (CONFIG["length_cm"] - np.abs(u_cm)) *
            (CONFIG["length_cm"] - np.abs(v_cm))
        ) / 10000.0

        keep = (
            (theta_deg <= CONFIG["theta_max_deg"]) &
            (overlap_m2 > 0)
        )
        if not np.any(keep):
            continue

        dx = dx[keep].astype(int)
        dy = dy[keep].astype(int)
        u_cm = u_cm[keep]
        v_cm = v_cm[keep]
        overlap_m2 = overlap_m2[keep]
        t_valid = t[valid].to_numpy()[keep]

        eff_weight = (
            W1[iX1_raw[keep], iY1_raw[keep]] *
            W2[iX2_raw[keep], iY2_raw[keep]]
        ).astype(float)

        half = 0.5 * CONFIG["pitch_cm"]
        omega_sr = np.array([
            rect_solid_angle(
                (u - half) / 100.0, (u + half) / 100.0,
                (v - half) / 100.0, (v + half) / 100.0,
                CONFIG["distance_cm"] / 100.0
            )
            for u, v in zip(u_cm, v_cm)
        ], dtype=float)

        keep2 = omega_sr > 0
        if not np.any(keep2):
            continue

        dx = dx[keep2]
        dy = dy[keep2]
        eff_weight = eff_weight[keep2]
        overlap_m2 = overlap_m2[keep2]
        omega_sr = omega_sr[keep2]
        t_valid = t_valid[keep2]

        geom_factor = 1.0 / (overlap_m2 * omega_sr)
        corr_numerator = eff_weight * geom_factor

        out = pd.DataFrame({
            "time": t_valid,
            "delta_x": dx,
            "delta_y": dy,
            "raw_increment": np.ones(len(dx), dtype=float),
            "eff_weight": eff_weight,
            "geom_factor": geom_factor,
            "corr_numerator": corr_numerator,
            "corr_numerator_sq": corr_numerator**2,
        })
        rows.append(out)

    if not rows:
        raise ValueError("No valid 4-plane events survived reconstruction and geometric cuts.")

    events = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    events["dt_s"] = events["time"].diff().dt.total_seconds()
    events["live_dt_s"] = np.where(
        events["dt_s"].fillna(np.inf).to_numpy() <= CONFIG["live_threshold_s"],
        events["dt_s"].fillna(0.0).to_numpy(),
        0.0,
    )

    meta = {
        "input_type": "events",
        "total_rows": total_rows,
        "valid_events_4planes": valid_events,
        "multi_hit_events_within_valid": multi_hit_events,
        "t_start": str(events["time"].iloc[0]),
        "t_end": str(events["time"].iloc[-1]),
        "span_s": float((events["time"].iloc[-1] - events["time"].iloc[0]).total_seconds()),
        "live_threshold_s": float(CONFIG["live_threshold_s"]),
        "live_time_s": float(events["live_dt_s"].sum()),
        "removed_gaps": int((events["dt_s"].fillna(np.inf) > CONFIG["live_threshold_s"]).sum()),
        "removed_time_s": float(events.loc[events["dt_s"].fillna(np.inf) > CONFIG["live_threshold_s"], "dt_s"].fillna(0.0).sum()),
    }
    return events, meta


def read_delta_map(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "counts" not in df.columns:
        for alt in ["count", "value", "n", "conteos"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "counts"})
                break
    need = {"delta_x", "delta_y", "counts"}
    if not need.issubset(df.columns):
        raise ValueError(f"The delta-map must contain {sorted(need)}")
    out = df[["delta_x", "delta_y", "counts"]].copy()
    out["delta_x"] = out["delta_x"].astype(int)
    out["delta_y"] = out["delta_y"].astype(int)
    out["counts"] = out["counts"].astype(float)
    meta = {
        "input_type": "map",
        "note": (
            "GIFs skipped: a delta-map has no time axis. "
            "Efficiency correction per event cannot be reconstructed exactly; "
            "only geometric correction is applied and the error map is approximate."
        ),
    }
    return out.sort_values(["delta_y", "delta_x"]).reset_index(drop=True), meta


def build_bin_edges(times: pd.Series, step_s: float) -> pd.DatetimeIndex:
    if step_s <= 0:
        raise ValueError("CONFIG['step_s'] must be > 0.")
    t0 = times.iloc[0]
    t1 = times.iloc[-1]
    n_steps = max(1, int(np.ceil((t1 - t0).total_seconds() / step_s)))
    edges = [t0 + pd.Timedelta(seconds=i * step_s) for i in range(n_steps + 1)]
    if edges[-1] < t1:
        edges.append(t1)
    return pd.DatetimeIndex(edges)


def delta_edges_to_theta_edges(max_delta: int) -> tuple[np.ndarray, np.ndarray]:
    delta_edges = np.arange(-max_delta - 0.5, max_delta + 1.5, 1.0)
    theta_edges_deg = np.degrees(np.arctan(delta_edges * (CONFIG["pitch_cm"] / CONFIG["distance_cm"])))
    theta_x_edges = theta_edges_deg
    theta_y_edges_global = theta_edges_deg + CONFIG["tilt_y_deg"]
    X, Y = np.meshgrid(theta_x_edges, theta_y_edges_global)
    return X, Y


def build_matrix_from_dict(counts: dict[tuple[int, int], float], xs: np.ndarray, ys: np.ndarray) -> np.ma.MaskedArray:
    mat = np.full((len(ys), len(xs)), np.nan, dtype=float)
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}
    for (dx, dy), value in counts.items():
        if dx in xi and dy in yi:
            mat[yi[dy], xi[dx]] = float(value)
    return np.ma.masked_invalid(mat)


def robust_limits(values: np.ndarray, pmin: float, pmax: float) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, pmin))
    hi = float(np.percentile(vals, pmax))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def choose_scale(values: np.ndarray, key: str) -> tuple[float, float]:
    cfg = CONFIG["color_scales"][key]
    if cfg["mode"] == "fixed":
        vmin = cfg["vmin"]
        vmax = cfg["vmax"]
        if vmax is None:
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]
            vmax = float(np.nanmax(vals)) if vals.size else (float(vmin) + 1.0)
        return float(vmin), float(vmax)

    pmin, pmax = cfg["robust_percentiles"]
    auto_lo, auto_hi = robust_limits(values, pmin, pmax)
    vmin = auto_lo if cfg["vmin"] is None else float(cfg["vmin"])
    vmax = auto_hi if cfg["vmax"] is None else float(cfg["vmax"])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def get_cmap(name: str):
    cmap = colormaps[name].copy()
    cmap.set_bad("white")
    return cmap


def render_map(
    mat: np.ma.MaskedArray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    out_png: Path,
    title: str,
    subtitle: str,
    cbar_label: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        mat,
        shading="auto",
        cmap=get_cmap(cmap_name),
        norm=Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12)),
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    ax.set_xlabel(r"$\theta_x$ [deg]")
    ax.set_ylabel(r"$\theta_{y,\mathrm{global}}$ [deg]")
    ax.set_title(title, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(*CONFIG["ylim_global_deg"])
    ax.grid(False)
    ax.text(
        0.015, 0.985, subtitle,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=CONFIG["annot_fontsize"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.90, edgecolor="0.75"),
    )
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=CONFIG["gif_frame_dpi"] if "frame_" in out_png.name else CONFIG["png_dpi"], bbox_inches="tight")
    plt.close(fig)


def safe_relative_error(sum_w: dict[tuple[int, int], float], sum_w2: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    out = {}
    for key, num in sum_w.items():
        if num > 0:
            out[key] = float(np.sqrt(sum_w2.get(key, 0.0)) / num) * 100.0
    return out


def safe_absolute_error(sum_w2: dict[tuple[int, int], float], denom: float) -> dict[tuple[int, int], float]:
    if denom <= 0:
        return {}
    return {key: float(np.sqrt(val) / denom) for key, val in sum_w2.items() if val > 0}


def save_gif(frame_paths: list[Path], out_path: Path) -> None:
    duration_ms = int(round(1000 / max(int(CONFIG["gif_fps"]), 1)))
    images = []
    for p in frame_paths:
        with Image.open(p) as im:
            images.append(im.convert("P", palette=Image.Palette.ADAPTIVE))
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )


def final_event_maps(events: pd.DataFrame) -> tuple[dict, dict, dict, float]:
    raw_counts = events.groupby(["delta_x", "delta_y"])["raw_increment"].sum().to_dict()
    corr_num = events.groupby(["delta_x", "delta_y"])["corr_numerator"].sum().to_dict()
    corr_num_sq = events.groupby(["delta_x", "delta_y"])["corr_numerator_sq"].sum().to_dict()
    live_time = float(events["live_dt_s"].sum())

    if live_time > 0:
        corrected_map = {k: v / live_time for k, v in corr_num.items()}
    else:
        corrected_map = corr_num.copy()

    if CONFIG["error_kind"] == "relative":
        error_map = safe_relative_error(corr_num, corr_num_sq)
    else:
        error_map = safe_absolute_error(corr_num_sq, live_time)

    return raw_counts, corrected_map, error_map, live_time


def final_map_from_delta(delta_map: pd.DataFrame) -> tuple[dict, dict, dict, float]:
    raw_counts = {
        (int(dx), int(dy)): float(c)
        for dx, dy, c in delta_map[["delta_x", "delta_y", "counts"]].itertuples(index=False)
    }

    corrected_map = {}
    error_map = {}

    half = 0.5 * CONFIG["pitch_cm"]
    for dx, dy, counts in delta_map[["delta_x", "delta_y", "counts"]].itertuples(index=False):
        u_cm = float(dx) * CONFIG["pitch_cm"]
        v_cm = float(dy) * CONFIG["pitch_cm"]
        theta_deg = float(np.degrees(np.arctan2(np.sqrt(u_cm**2 + v_cm**2), CONFIG["distance_cm"])))
        overlap_m2 = ((CONFIG["length_cm"] - abs(u_cm)) * (CONFIG["length_cm"] - abs(v_cm))) / 10000.0
        if theta_deg > CONFIG["theta_max_deg"] or overlap_m2 <= 0:
            continue
        omega_sr = rect_solid_angle(
            (u_cm - half) / 100.0, (u_cm + half) / 100.0,
            (v_cm - half) / 100.0, (v_cm + half) / 100.0,
            CONFIG["distance_cm"] / 100.0
        )
        if omega_sr <= 0:
            continue
        geom_factor = 1.0 / (overlap_m2 * omega_sr)
        corrected = float(counts) * geom_factor
        corrected_map[(int(dx), int(dy))] = corrected

        if CONFIG["error_kind"] == "relative":
            if counts > 0:
                error_map[(int(dx), int(dy))] = 100.0 / np.sqrt(float(counts))
        else:
            error_map[(int(dx), int(dy))] = np.sqrt(float(counts)) * geom_factor

    return raw_counts, corrected_map, error_map, float("nan")


def write_summary(summary_path: Path, meta: dict, scales: dict[str, tuple[float, float]], input_type: str, outdir: Path) -> None:
    lines = []
    lines.append("Unified muography summary")
    lines.append("=" * 80)
    lines.append(f"Input type: {input_type}")
    lines.append(f"Output directory: {outdir}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 80)
    for key in [
        "pitch_cm", "length_cm", "distance_cm", "tilt_y_deg", "theta_max_deg",
        "step_s", "gif_fps", "gif_mode", "live_threshold_s", "error_kind"
    ]:
        lines.append(f"{key}: {CONFIG[key]}")
    lines.append("")
    lines.append("Colormaps")
    lines.append("-" * 80)
    for key, name in CONFIG["colormaps"].items():
        lo, hi = scales[key]
        lines.append(f"{key}: cmap={name}, vmin={lo:.6g}, vmax={hi:.6g}")
    lines.append("")
    lines.append("Metadata")
    lines.append("-" * 80)
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def export_final_tables(outdir: Path, raw_counts: dict, corrected_map: dict, error_map: dict) -> None:
    keys = sorted(set(raw_counts) | set(corrected_map) | set(error_map))
    rows = []
    for dx, dy in keys:
        rows.append({
            "delta_x": dx,
            "delta_y": dy,
            "raw_counts": raw_counts.get((dx, dy), np.nan),
            "corrected_value": corrected_map.get((dx, dy), np.nan),
            "error_value": error_map.get((dx, dy), np.nan),
        })
    pd.DataFrame(rows).to_csv(outdir / "final_maps_table.csv", index=False)


def make_gifs_and_final_pngs(events: pd.DataFrame, outdir: Path) -> dict[str, tuple[float, float]]:
    outdir.mkdir(parents=True, exist_ok=True)

    raw_frames_dir = outdir / "raw_frames"
    corr_frames_dir = outdir / "corrected_frames"
    err_frames_dir = outdir / "error_frames"
    if CONFIG["keep_frame_pngs"]:
        raw_frames_dir.mkdir(parents=True, exist_ok=True)
        corr_frames_dir.mkdir(parents=True, exist_ok=True)
        err_frames_dir.mkdir(parents=True, exist_ok=True)

    raw_final_counts, corr_final_map, err_final_map, total_live_time = final_event_maps(events)

    max_delta = int(max(np.abs(events["delta_x"]).max(), np.abs(events["delta_y"]).max()))
    xs = np.arange(-max_delta, max_delta + 1)
    ys = np.arange(-max_delta, max_delta + 1)
    x_edges, y_edges = delta_edges_to_theta_edges(max_delta)

    raw_scale = choose_scale(np.array(list(raw_final_counts.values()), dtype=float), "raw")
    corr_scale = choose_scale(np.array(list(corr_final_map.values()), dtype=float), "corrected")
    err_scale = choose_scale(np.array(list(err_final_map.values()), dtype=float), "error")
    scales = {"raw": raw_scale, "corrected": corr_scale, "error": err_scale}

    edges = build_bin_edges(events["time"], CONFIG["step_s"])
    events = events.copy()
    events["bin_id"] = pd.cut(events["time"], bins=edges, labels=False, include_lowest=True, right=True)

    raw_cumulative_counts = {}
    corr_cumulative_num = {}
    corr_cumulative_num_sq = {}
    cumulative_live_time = 0.0

    raw_frame_paths = []
    corr_frame_paths = []
    err_frame_paths = []
    summary_rows = []

    for frame_idx in range(len(edges) - 1):
        left = edges[frame_idx]
        right = edges[frame_idx + 1]

        chunk = events.loc[
            events["bin_id"] == frame_idx,
            ["delta_x", "delta_y", "corr_numerator", "corr_numerator_sq", "live_dt_s"]
        ]
        n_events_frame = int(len(chunk))
        live_time_bin = float(chunk["live_dt_s"].sum())

        if CONFIG["gif_mode"] == "incremental":
            raw_current_counts = {}
            corr_current_num = {}
            corr_current_num_sq = {}
            for dx, dy, corr_num, corr_num_sq, _ in chunk.itertuples(index=False):
                key = (int(dx), int(dy))
                raw_current_counts[key] = raw_current_counts.get(key, 0.0) + 1.0
                corr_current_num[key] = corr_current_num.get(key, 0.0) + float(corr_num)
                corr_current_num_sq[key] = corr_current_num_sq.get(key, 0.0) + float(corr_num_sq)
            denom = live_time_bin
        else:
            for dx, dy, corr_num, corr_num_sq, live_dt in chunk.itertuples(index=False):
                key = (int(dx), int(dy))
                raw_cumulative_counts[key] = raw_cumulative_counts.get(key, 0.0) + 1.0
                corr_cumulative_num[key] = corr_cumulative_num.get(key, 0.0) + float(corr_num)
                corr_cumulative_num_sq[key] = corr_cumulative_num_sq.get(key, 0.0) + float(corr_num_sq)
                cumulative_live_time += float(live_dt)
            raw_current_counts = raw_cumulative_counts
            corr_current_num = corr_cumulative_num
            corr_current_num_sq = corr_cumulative_num_sq
            denom = cumulative_live_time

        if denom > 0:
            corr_current = {k: v / denom for k, v in corr_current_num.items()}
            corr_label = r"Corrected flux [m$^{-2}$ sr$^{-1}$ s$^{-1}$]"
            time_note = f"live time shown: {denom:.1f} s"
        else:
            corr_current = corr_current_num.copy()
            corr_label = r"Corrected intensity [m$^{-2}$ sr$^{-1}$]"
            time_note = "live time shown: 0.0 s (unnormalized numerator)"

        if CONFIG["error_kind"] == "relative":
            err_current = safe_relative_error(corr_current_num, corr_current_num_sq)
            err_label = r"1$\sigma$ relative uncertainty [%]"
            err_title = "Corrected map statistical uncertainty"
        else:
            err_current = safe_absolute_error(corr_current_num_sq, denom)
            err_label = r"1$\sigma$ absolute uncertainty [m$^{-2}$ sr$^{-1}$ s$^{-1}$]"
            err_title = "Corrected map absolute statistical uncertainty"

        raw_mat = build_matrix_from_dict(raw_current_counts, xs, ys)
        corr_mat = build_matrix_from_dict(corr_current, xs, ys)
        err_mat = build_matrix_from_dict(err_current, xs, ys)
        n_events_shown = int(sum(raw_current_counts.values()))

        raw_png = (raw_frames_dir / f"frame_{frame_idx:04d}.png") if CONFIG["keep_frame_pngs"] else (outdir / f"_tmp_raw_{frame_idx:04d}.png")
        corr_png = (corr_frames_dir / f"frame_{frame_idx:04d}.png") if CONFIG["keep_frame_pngs"] else (outdir / f"_tmp_corr_{frame_idx:04d}.png")
        err_png = (err_frames_dir / f"frame_{frame_idx:04d}.png") if CONFIG["keep_frame_pngs"] else (outdir / f"_tmp_err_{frame_idx:04d}.png")

        subtitle = (
            f"{left.strftime('%Y-%m-%d %H:%M:%S')} → {right.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"events in frame: {n_events_frame} | events shown: {n_events_shown}\n"
            f"{time_note}"
        )

        render_map(
            mat=raw_mat,
            x_edges=x_edges,
            y_edges=y_edges,
            out_png=raw_png,
            title="Raw angular counts",
            subtitle=subtitle,
            cbar_label="Raw counts",
            cmap_name=CONFIG["colormaps"]["raw"],
            vmin=raw_scale[0],
            vmax=raw_scale[1],
        )
        render_map(
            mat=corr_mat,
            x_edges=x_edges,
            y_edges=y_edges,
            out_png=corr_png,
            title="Corrected angular map",
            subtitle=subtitle,
            cbar_label=corr_label,
            cmap_name=CONFIG["colormaps"]["corrected"],
            vmin=corr_scale[0],
            vmax=corr_scale[1],
        )
        render_map(
            mat=err_mat,
            x_edges=x_edges,
            y_edges=y_edges,
            out_png=err_png,
            title=err_title,
            subtitle=subtitle,
            cbar_label=err_label,
            cmap_name=CONFIG["colormaps"]["error"],
            vmin=err_scale[0],
            vmax=err_scale[1],
        )

        raw_frame_paths.append(raw_png)
        corr_frame_paths.append(corr_png)
        err_frame_paths.append(err_png)

        summary_rows.append({
            "frame": frame_idx,
            "t_start": left,
            "t_end": right,
            "events_in_frame": n_events_frame,
            "events_shown": n_events_shown,
            "live_time_frame_s": live_time_bin,
            "live_time_shown_s": denom,
        })

    if not raw_frame_paths:
        raise ValueError("No GIF frames were generated.")

    save_gif(raw_frame_paths, outdir / "raw_counts_fill.gif")
    save_gif(corr_frame_paths, outdir / "corrected_fill.gif")
    save_gif(err_frame_paths, outdir / "error_fill.gif")
    pd.DataFrame(summary_rows).to_csv(outdir / "frame_summary.csv", index=False)

    # Final PNG maps from the full dataset
    raw_final_mat = build_matrix_from_dict(raw_final_counts, xs, ys)
    corr_final_mat = build_matrix_from_dict(corr_final_map, xs, ys)
    err_final_mat = build_matrix_from_dict(err_final_map, xs, ys)

    final_note = (
        f"full integration\n"
        f"events shown: {int(sum(raw_final_counts.values()))}\n"
        f"live time: {total_live_time:.1f} s"
    )

    render_map(
        mat=raw_final_mat,
        x_edges=x_edges,
        y_edges=y_edges,
        out_png=outdir / "01_raw_counts_final.png",
        title="Raw angular counts",
        subtitle=final_note,
        cbar_label="Raw counts",
        cmap_name=CONFIG["colormaps"]["raw"],
        vmin=raw_scale[0],
        vmax=raw_scale[1],
    )
    render_map(
        mat=corr_final_mat,
        x_edges=x_edges,
        y_edges=y_edges,
        out_png=outdir / "02_corrected_map_final.png",
        title="Corrected angular map",
        subtitle=final_note,
        cbar_label=r"Corrected flux [m$^{-2}$ sr$^{-1}$ s$^{-1}$]" if total_live_time > 0 else r"Corrected intensity [m$^{-2}$ sr$^{-1}$]",
        cmap_name=CONFIG["colormaps"]["corrected"],
        vmin=corr_scale[0],
        vmax=corr_scale[1],
    )
    render_map(
        mat=err_final_mat,
        x_edges=x_edges,
        y_edges=y_edges,
        out_png=outdir / "03_error_map_final.png",
        title="Corrected map statistical uncertainty" if CONFIG["error_kind"] == "relative" else "Corrected map absolute statistical uncertainty",
        subtitle=final_note,
        cbar_label=(r"1$\sigma$ relative uncertainty [%]" if CONFIG["error_kind"] == "relative"
                    else r"1$\sigma$ absolute uncertainty [m$^{-2}$ sr$^{-1}$ s$^{-1}$]"),
        cmap_name=CONFIG["colormaps"]["error"],
        vmin=err_scale[0],
        vmax=err_scale[1],
    )

    if not CONFIG["keep_frame_pngs"]:
        for p in raw_frame_paths + corr_frame_paths + err_frame_paths:
            try:
                p.unlink()
            except OSError:
                pass

    export_final_tables(outdir, raw_final_counts, corr_final_map, err_final_map)
    return scales


def make_static_pngs_from_delta(delta_map: pd.DataFrame, outdir: Path) -> dict[str, tuple[float, float]]:
    outdir.mkdir(parents=True, exist_ok=True)
    raw_counts, corrected_map, error_map, _ = final_map_from_delta(delta_map)

    max_delta = int(max(np.abs(delta_map["delta_x"]).max(), np.abs(delta_map["delta_y"]).max()))
    xs = np.arange(-max_delta, max_delta + 1)
    ys = np.arange(-max_delta, max_delta + 1)
    x_edges, y_edges = delta_edges_to_theta_edges(max_delta)

    raw_scale = choose_scale(np.array(list(raw_counts.values()), dtype=float), "raw")
    corr_scale = choose_scale(np.array(list(corrected_map.values()), dtype=float), "corrected")
    err_scale = choose_scale(np.array(list(error_map.values()), dtype=float), "error")
    scales = {"raw": raw_scale, "corrected": corr_scale, "error": err_scale}

    raw_mat = build_matrix_from_dict(raw_counts, xs, ys)
    corr_mat = build_matrix_from_dict(corrected_map, xs, ys)
    err_mat = build_matrix_from_dict(error_map, xs, ys)

    subtitle = "delta-map input\nno time axis available\nGIFs skipped"

    render_map(
        raw_mat, x_edges, y_edges, outdir / "01_raw_counts_final.png",
        "Raw angular counts", subtitle, "Raw counts",
        CONFIG["colormaps"]["raw"], raw_scale[0], raw_scale[1]
    )
    render_map(
        corr_mat, x_edges, y_edges, outdir / "02_corrected_map_final.png",
        "Geometry-corrected angular map", subtitle,
        r"Corrected intensity [m$^{-2}$ sr$^{-1}$]",
        CONFIG["colormaps"]["corrected"], corr_scale[0], corr_scale[1]
    )
    render_map(
        err_mat, x_edges, y_edges, outdir / "03_error_map_final.png",
        "Approximate statistical uncertainty", subtitle,
        (r"1$\sigma$ relative uncertainty [%]" if CONFIG["error_kind"] == "relative"
         else r"1$\sigma$ absolute uncertainty [m$^{-2}$ sr$^{-1}$]"),
        CONFIG["colormaps"]["error"], err_scale[0], err_scale[1]
    )
    export_final_tables(outdir, raw_counts, corrected_map, error_map)
    return scales


def main() -> None:
    setup_style()

    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage:\n  python muography_unified_article_gif.py <input.csv>")
        sys.exit(0 if len(sys.argv) == 2 else 1)

    input_path = Path(sys.argv[1]).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    outdir = Path(CONFIG["output_dir"]) if CONFIG["output_dir"] else input_path.with_name(input_path.stem + "_muography_output")
    input_type = detect_input_type(input_path)

    if input_type == "events":
        events, meta = reconstruct_event_table(input_path)
        scales = make_gifs_and_final_pngs(events, outdir)
    else:
        delta_map, meta = read_delta_map(input_path)
        scales = make_static_pngs_from_delta(delta_map, outdir)

    write_summary(outdir / "summary.txt", meta, scales, input_type, outdir)

    print(f"Done. Output directory: {outdir}")
    if input_type == "events":
        print(f"  - {outdir / 'raw_counts_fill.gif'}")
        print(f"  - {outdir / 'corrected_fill.gif'}")
        print(f"  - {outdir / 'error_fill.gif'}")
        print(f"  - {outdir / '01_raw_counts_final.png'}")
        print(f"  - {outdir / '02_corrected_map_final.png'}")
        print(f"  - {outdir / '03_error_map_final.png'}")
    else:
        print("GIFs were skipped because the input has no time axis.")
        print(f"  - {outdir / '01_raw_counts_final.png'}")
        print(f"  - {outdir / '02_corrected_map_final.png'}")
        print(f"  - {outdir / '03_error_map_final.png'}")


if __name__ == "__main__":
    main()
