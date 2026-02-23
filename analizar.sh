#!/usr/bin/env bash
set -euo pipefail

# Ejecuta scripts de ./modulos sobre un *_coinc4.csv ya generado.
# Uso:
#   ./run_from_coinc4.sh /ruta/al/archivo_coinc4.csv

if [[ $# -ne 1 ]]; then
  echo "Uso: $0 /ruta/al/archivo_coinc4.csv" >&2
  exit 1
fi

CSV="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODDIR="${REPO_ROOT}/modulos"
PYTHON="${PYTHON:-python3}"

[[ -f "$CSV" ]] || { echo "ERROR: No existe el archivo: $CSV" >&2; exit 1; }

echo "Archivo: $CSV"
echo "Modulos: $MODDIR"
echo "Python : $PYTHON"
echo

echo "=== analisis_global.py ==="
"$PYTHON" "$MODDIR/analisis_global.py" "$CSV"
echo

echo "=== flujo_4fold.py ==="
"$PYTHON" "$MODDIR/flujo_4fold.py" "$CSV"
echo

echo "=== histograma_4fold.py ==="
"$PYTHON" "$MODDIR/histograma_4fold.py" "$CSV"
echo

# echo "=== angulo_derecho.py ==="
# "$PYTHON" "$MODDIR/angulo_derecho.py" "$CSV"
# echo

echo "=== angulo_volteado.py ==="
"$PYTHON" "$MODDIR/angulo_volteado.py" "$CSV"
echo

echo "OK: pipeline terminado."
