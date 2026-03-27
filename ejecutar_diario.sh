#!/usr/bin/env bash
set -euo pipefail

BASE="/home/simulaciones/rafael/MuTe_Instrumento"
ACUM="$BASE/datos/Acumulado_Montaña"

FECHA_YMD=$(date -d "yesterday" +%Y%m%d)
FECHA_DMY=$(date -d "yesterday" +%d%m%Y)

OUTDIR="$BASE/datos/$FECHA_DMY"
CSV="$OUTDIR/archivo_concatenado_coinc4.csv"
LOG="$BASE/logs/cron_${FECHA_YMD}.log"

mkdir -p "$BASE/logs" "$OUTDIR" "$ACUM"

cd "$BASE"

./bajar.sh \
  --date "$FECHA_YMD" \
  --scriptsdir "$BASE/modulos" \
  --outdir "$OUTDIR" >> "$LOG" 2>&1

./analizar.sh "$CSV" >> "$LOG" 2>&1

shopt -s nullglob
files=("$OUTDIR"/MUTE_MACH1_*as_MuTe.csv)
if ((${#files[@]})); then
  cp -f "${files[@]}" "$ACUM"/
else
  echo "No se encontraron archivos MUTE_MACH1_*as_MuTe.csv en $OUTDIR" >> "$LOG"
fi

rm -rf "$ACUM"/archivo_concatenado*

python3 "$BASE/modulos/unircsv.py" "$ACUM/" >> "$LOG" 2>&1
python3 "$BASE/modulos/filtro_coincidencias.py" "$ACUM/archivo_concatenado.csv" >> "$LOG" 2>&1

sleep 300

./analizar.sh "$ACUM/archivo_concatenado_coinc4.csv" >> "$LOG" 2>&1
