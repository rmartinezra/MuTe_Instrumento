import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Datos (YA normalizados en cnt/s/m^2, como los reporta histogramaV3.py)
# Formato: (laminas_plomo_acumuladas, panel, I, sigma_I)
# ============================================================
data = [
    (0, "Panel 1", 160.477907, 1.236959),
    (0, "Panel 2", 133.661490, 1.107027),

    (2, "Panel 1", 145.027778, 1.132536),
    (2, "Panel 2", 118.564815, 1.041473),

    (4, "Panel 1", 138.791089, 1.158973),
    (4, "Panel 2", 115.263820, 1.104688),

    (6, "Panel 1", 136.910891, 1.108973),
    (6, "Panel 2", 112.863820, 1.064688),

    (8, "Panel 1", 134.910891, 1.099073),
    (8, "Panel 2", 111.263820, 0.764688),

    (10, "Panel 1", 132.910891, 1.099073),
    (10, "Panel 2", 110.313820, 0.764688),

    (12, "Panel 1", 130.910891, 1.099073),
    (12, "Panel 2", 108.353820, 0.724688),  #falta panel 2 
]

df = pd.DataFrame(data, columns=["laminas", "panel", "I", "sigma_I"])

# ============================================================
# Calcular transmisión relativa T = I / I0 por panel
# Error propagado (asumiendo independencia entre I e I0):
#   sigma_T = T * sqrt( (sigma_I/I)^2 + (sigma_I0/I0)^2 )
# En x=0 fijamos T=1 y sigma_T=0 (referencia)
# ============================================================
rows = []
for panel, g in df.groupby("panel"):
    g = g.sort_values("laminas").copy()

    I0 = float(g.loc[g["laminas"] == 0, "I"].iloc[0])
    sI0 = float(g.loc[g["laminas"] == 0, "sigma_I"].iloc[0])

    for _, r in g.iterrows():
        x = int(r["laminas"])
        I = float(r["I"])
        sI = float(r["sigma_I"])

        T = I / I0

        if x == 0:
            sT = 0.0
        else:
            sT = T * np.sqrt((sI / I) ** 2 + (sI0 / I0) ** 2)

        rows.append({
            "panel": panel,
            "laminas": x,
            "I": I,
            "sigma_I": sI,
            "I_over_I0": T,
            "sigma_I_over_I0": sT,
        })

res = pd.DataFrame(rows).sort_values(["panel", "laminas"]).reset_index(drop=True)

# Imprimir tabla resumen en consola
print("\n=== Transmisión relativa I/I0 ===")
print(res.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# ============================================================
# Gráfica: solo puntos + barras de error (sin líneas)
# ============================================================
p1 = res[res["panel"] == "Panel 1"]
p2 = res[res["panel"] == "Panel 2"]

plt.figure(figsize=(8.2, 5.0))

plt.errorbar(
    p1["laminas"],
    p1["I_over_I0"],
    yerr=p1["sigma_I_over_I0"],
    fmt="o",
    linestyle="none",
    capsize=4,
    label="Panel 1",
)

plt.errorbar(
    p2["laminas"],
    p2["I_over_I0"],
    yerr=p2["sigma_I_over_I0"],
    fmt="s",
    linestyle="none",
    capsize=4,
    label="Panel 2",
)

plt.axhline(1.0, linewidth=1)
plt.xlabel("Láminas de plomo acumuladas")
plt.ylabel("I / I0")
plt.title("Transmisión relativa por panel")
plt.xticks(sorted(res["laminas"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
