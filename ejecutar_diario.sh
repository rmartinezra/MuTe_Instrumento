#!/usr/bin/env bash
set -euo pipefail

BASE="/home/simulaciones/rafael/MuTe_Instrumento"
PASSFILE="$HOME/.mute_sshpass"

FECHA_YMD=$(date -d "yesterday" +%Y%m%d)   # 20260325
FECHA_DMY=$(date -d "yesterday" +%d%m%Y)   # 25032026

LOGDIR="$BASE/logs"
OUTDIR="$BASE/datos/$FECHA_DMY"
CSV="$OUTDIR/archivo_concatenado_coinc4.csv"

mkdir -p "$LOGDIR" "$OUTDIR"

if [[ ! -f "$PASSFILE" ]]; then
  echo "No existe el archivo de clave: $PASSFILE" >&2
  exit 1
fi

export SSHPASS="$(<"$PASSFILE")"

cd "$BASE"

./bajar.sh \
  --date "$FECHA_YMD" \
  --scriptsdir "$BASE/modulos" \
  --outdir "$OUTDIR" \
  >> "$LOGDIR/bajar_${FECHA_YMD}.log" 2>&1

if [[ -f "$CSV" ]]; then
  ./analizar.sh "$CSV" >> "$LOGDIR/analizar_${FECHA_YMD}.log" 2>&1
else
  echo "No se encontró $CSV" >> "$LOGDIR/analizar_${FECHA_YMD}.log"
  exit 1
fi
