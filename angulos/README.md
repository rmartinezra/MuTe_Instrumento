# MuTe / ARTI angular workflow — README

Este repositorio contiene scripts para:
1) transformar histogramas angulares de ARTI a cantidades comparables con flujo diferencial,
2) modelar la aceptancia geométrica de un telescopio de dos planos cuadrados tipo MuTe,
3) corregir mapas medidos por aceptancia (y, si hace falta, por un término residual ε),
4) comparar aceptancia analítica vs simulada (Monte Carlo ray-tracing),
5) estudiar el efecto de incluir o no el factor cosθ (convención de área).

---

## Estructura recomendada

project/
- main.tex                       (documento Overleaf)
- README.md
- data/
  - bga_3600.csv                 (salida ARTI con columna theta en grados)
  - mapa_pixeles_delta_xy.csv    (mapa de coincidencias: delta_x, delta_y, counts)
  - (outputs .csv de diagnóstico se guardan aquí)
- code/
  - 01_arti_histograms.py
  - 02_mute_corrections.py
  - 03_acceptance_mc_vs_analytic.py
  - 04_acceptance_with_without_cos.py
- figs/
  - (todas las figuras .png generadas por los scripts)

---

## Requisitos

Python 3.9+ y paquetes:
- numpy
- pandas
- matplotlib

Instalación típica:
```bash
pip install numpy pandas matplotlib
