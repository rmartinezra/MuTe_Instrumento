#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
shopt -s nullglob

# -----------------------
# Defaults
# -----------------------
USER_HOST="martinezr@10.1.157.94"
REMOTE_DIR="/share/CACHEDEV1_DATA/data/mute"
PREFIX="MUTE_MACH1_"
OUTDIR="."
SCRIPTS_DIR=""
TRADUCCION_SCRIPT=""
DATE_STR=""

PYTHON_BIN="python3"
SSH_PORT=""
IDENTITY_FILE=""
RETRIES=2
KEEP_ARCHIVES=1
FORCE=0
SKIP_DOWNLOAD=0
SKIP_EXTRACT=0
SKIP_PYTHON=0
DRY_RUN=0

# -----------------------
# Utils
# -----------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }

fmt_time() {
  local s="$1"
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local sec=$((s%60))
  printf "%02d:%02d:%02d" "$h" "$m" "$sec"
}

die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

on_err() {
  local lineno="${1:-?}"
  local rc="${2:-1}"
  local cmd="${3:-<unknown>}"
  echo "[$(ts)] ERROR: fallo en línea ${lineno}: ${cmd} (rc=${rc})" >&2
  exit "$rc"
}
trap 'on_err "$LINENO" "$?" "$BASH_COMMAND"' ERR

run_step() {
  local title="$1"; shift
  echo "[$(ts)] ==> $title"
  local t0=$SECONDS
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$(ts)] [DRY-RUN] $*"
    echo "[$(ts)] <== OK: $title (elapsed 00:00:00)"
    return 0
  fi
  "$@"
  local dt=$((SECONDS - t0))
  echo "[$(ts)] <== OK: $title (elapsed $(fmt_time "$dt"))"
}

# -----------------------
# CSV translation helpers
# -----------------------
translate_list_csvs_in_cwd() {
  # Traduce todos los *_list.csv (shaping) a *_as_MuTe.csv (wide ch00..ch63)
  shopt -s nullglob
  local files=( *_list.csv )
  if (( ${#files[@]} == 0 )); then
    echo "[$(ts)] No hay *_list.csv para traducir en $(pwd)"
    return 0
  fi
  for f in "${files[@]}"; do
    [[ "$f" == *_as_MuTe.csv ]] && continue
    echo "[$(ts)]   -> traduccionMuTe.py: $f"
    "$PYTHON_BIN" "$TRADUCCION_SCRIPT" "$f"
  done
}

delete_list_csvs_in_cwd() {
  # Borra solo los CSV originales en formato *_list.csv (los shaping)
  shopt -s nullglob
  local files=( *_list.csv )
  if (( ${#files[@]} == 0 )); then
    echo "[$(ts)] No hay *_list.csv para borrar en $(pwd)"
    return 0
  fi
  rm -f -- "${files[@]}"
}


# -----------------------
# CSV ordering helpers
# -----------------------
sort_csv_by_time_in_place() {
  # Ordena por la primera columna (timestamp ISO en 'time') preservando header.
  # Para formato YYYY-MM-DD HH:MM:SS.sssssssss, orden lexicográfico == orden temporal.
  local f="$1"
  [[ -f "$f" ]] || die "No existe CSV para ordenar: $f"

  local dir base tmp
  dir="$(dirname -- "$f")"
  base="$(basename -- "$f")"
  tmp="$dir/.${base}.tmp_sort"

  # Preserva header y ordena resto por columna 1 separada por coma
  {
    head -n 1 -- "$f"
    tail -n +2 -- "$f" | LC_ALL=C sort -t',' -k1,1
  } > "$tmp"

  [[ -s "$tmp" ]] || die "El ordenamiento produjo archivo vacío: $tmp"
  mv -f -- "$tmp" "$f"
}

usage() {
  cat <<EOF
Uso:
  $(basename "$0") --date YYYYMMDD --scriptsdir PATH [opciones]

Requeridos:
  --date YYYYMMDD
  --scriptsdir PATH   (donde están unircsv.py, traduccionMuTe.py y filtro_coincidencias.py)

Opciones:
  --outdir PATH
  --userhost USER@HOST
  --remotedir PATH
  --prefix STR
  --python PATH
  --traduccion PATH   (ruta a traduccionMuTe.py; defecto: scriptsdir/traduccionMuTe.py)
  --port N
  --identity PATH
  --retries N
  --keep-archives | --no-keep-archives
  --force
  --skip-download
  --skip-extract
  --skip-python
  --dry-run
  -h|--help
EOF
}

# -----------------------
# Parse args
# -----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)        DATE_STR="${2:-}"; shift 2 ;;
    --scriptsdir)  SCRIPTS_DIR="${2:-}"; shift 2 ;;
    --traduccion) TRADUCCION_SCRIPT="${2:-}"; shift 2 ;;
    --outdir)      OUTDIR="${2:-}"; shift 2 ;;
    --userhost)    USER_HOST="${2:-}"; shift 2 ;;
    --remotedir)   REMOTE_DIR="${2:-}"; shift 2 ;;
    --prefix)      PREFIX="${2:-}"; shift 2 ;;
    --python)      PYTHON_BIN="${2:-}"; shift 2 ;;
    --port)        SSH_PORT="${2:-}"; shift 2 ;;
    --identity)    IDENTITY_FILE="${2:-}"; shift 2 ;;
    --retries)     RETRIES="${2:-}"; shift 2 ;;
    --keep-archives)    KEEP_ARCHIVES=1; shift ;;
    --no-keep-archives) KEEP_ARCHIVES=0; shift ;;
    --force)       FORCE=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-extract)  SKIP_EXTRACT=1; shift ;;
    --skip-python)   SKIP_PYTHON=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) die "Argumento desconocido: $1 (usa --help)" ;;
  esac
done

# Defaults dependientes
if [[ -z "${SCRIPTS_DIR}" ]]; then
  SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ -z "${TRADUCCION_SCRIPT}" ]]; then
  TRADUCCION_SCRIPT="$SCRIPTS_DIR/traduccionMuTe.py"
fi

# -----------------------
# Validations
# -----------------------
[[ -n "$DATE_STR" ]] || die "Debes pasar --date YYYYMMDD"
[[ "$DATE_STR" =~ ^[0-9]{8}$ ]] || die "--date debe ser YYYYMMDD (8 dígitos)."

[[ -d "$SCRIPTS_DIR" ]] || die "--scriptsdir no existe: $SCRIPTS_DIR"
[[ -f "$SCRIPTS_DIR/unircsv.py" ]] || die "No encuentro $SCRIPTS_DIR/unircsv.py"
[[ -f "$SCRIPTS_DIR/filtro_coincidencias.py" ]] || die "No encuentro $SCRIPTS_DIR/filtro_coincidencias.py"
[[ -f "$TRADUCCION_SCRIPT" ]] || die "No encuentro traduccionMuTe.py: $TRADUCCION_SCRIPT (usa --traduccion PATH)"

mkdir -p "$OUTDIR"

command -v scp >/dev/null 2>&1 || die "No encuentro scp."
command -v ssh >/dev/null 2>&1 || die "No encuentro ssh."
command -v tar >/dev/null 2>&1 || die "No encuentro tar."
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "No encuentro $PYTHON_BIN."

[[ "$RETRIES" =~ ^[0-9]+$ ]] || die "--retries debe ser entero >=0"
if [[ -n "$SSH_PORT" ]]; then [[ "$SSH_PORT" =~ ^[0-9]+$ ]] || die "--port debe ser entero"; fi
if [[ -n "$IDENTITY_FILE" ]]; then [[ -f "$IDENTITY_FILE" ]] || die "--identity no existe: $IDENTITY_FILE"; fi

# -----------------------
# Build SSH/SCP arrays (sshpass opcional)
# -----------------------
SSH_OPTS=(-o ConnectTimeout=12 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
SCP_OPTS=(-p)

if [[ -n "$SSH_PORT" ]]; then
  SSH_OPTS+=(-p "$SSH_PORT")
  SCP_OPTS+=(-P "$SSH_PORT")
fi
if [[ -n "$IDENTITY_FILE" ]]; then
  SSH_OPTS+=(-i "$IDENTITY_FILE")
  SCP_OPTS+=(-i "$IDENTITY_FILE")
fi

SSH_BIN=(ssh "${SSH_OPTS[@]}")
SCP_BIN=(scp "${SCP_OPTS[@]}")

if command -v sshpass >/dev/null 2>&1 && [[ -n "${SSHPASS:-}" ]]; then
  echo "[$(ts)] sshpass detectado y SSHPASS seteado: usando sshpass -e."
  SSH_BIN=(sshpass -e "${SSH_BIN[@]}")
  SCP_BIN=(sshpass -e "${SCP_BIN[@]}")
fi

PATTERN="${PREFIX}${DATE_STR}*.tar.xz"

echo "[$(ts)] Config:"
echo "  user@host    : $USER_HOST"
echo "  remote dir   : $REMOTE_DIR"
echo "  pattern name : $PATTERN"
echo "  scripts dir  : $SCRIPTS_DIR"
echo "  traduccion   : $TRADUCCION_SCRIPT"
echo "  outdir       : $(cd "$OUTDIR" && pwd)"
echo

# -----------------------
# Remote listing (CORREGIDO)
# -----------------------
remote_list_files() {
  "${SSH_BIN[@]}" "$USER_HOST" "bash -s" -- "$REMOTE_DIR" "$PREFIX" "$DATE_STR" <<'EOS'
set -euo pipefail
REMOTE_DIR="$1"
PREFIX="$2"
DATE_STR="$3"

if [[ ! -d "$REMOTE_DIR" ]]; then
  echo "__NO_REMOTE_DIR__"
  exit 3
fi

if command -v find >/dev/null 2>&1; then
  find "$REMOTE_DIR" -maxdepth 1 -type f -name "${PREFIX}${DATE_STR}*.tar.xz" -print
else
  ls -1 "$REMOTE_DIR"/"${PREFIX}${DATE_STR}"*.tar.xz 2>/dev/null || true
fi
EOS
}

remote_files=()
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo "[$(ts)] ==> Chequeando acceso remoto (ssh) y listando archivos"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$(ts)] [DRY-RUN] ssh list"
  else
    if out="$(remote_list_files)"; then
      :
    else
      rc=$?
      if [[ "$rc" -eq 3 ]]; then
        die "El directorio remoto no existe: $REMOTE_DIR"
      fi
      die "Falló ssh list (rc=$rc)"
    fi

    mapfile -t remote_files < <(printf "%s\n" "$out" | sed '/^$/d') || true
    if [[ "${#remote_files[@]}" -gt 0 && "${remote_files[0]}" == "__NO_REMOTE_DIR__" ]]; then
      die "El directorio remoto no existe: $REMOTE_DIR"
    fi
    [[ "${#remote_files[@]}" -gt 0 ]] || die "No hay archivos remotos para $DATE_STR (pattern: $PATTERN)"
    echo "[$(ts)] Archivos remotos encontrados: ${#remote_files[@]}"
  fi
else
  echo "[$(ts)] Descarga deshabilitada (--skip-download)."
fi

# -----------------------
# Download with retries
# -----------------------
downloaded=()

scp_one() {
  local remote_path="$1"
  local dest_dir="$2"
  local base dest
  base="$(basename "$remote_path")"
  dest="$dest_dir/$base"

  if [[ -f "$dest" && "$FORCE" -eq 0 ]]; then
    echo "[$(ts)] Ya existe local: $dest (usa --force para sobreescribir). Omito."
    downloaded+=("$dest")
    return 0
  fi

  local attempt=0
  while :; do
    attempt=$((attempt+1))
    echo "[$(ts)] scp [$attempt/$((RETRIES+1))]: $remote_path -> $dest_dir/"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      downloaded+=("$dest")
      return 0
    fi
    if "${SCP_BIN[@]}" "${USER_HOST}:${remote_path}" "$dest_dir/"; then
      [[ -s "$dest" ]] || die "Archivo descargado pero vacío: $dest"
      downloaded+=("$dest")
      return 0
    fi
    if [[ "$attempt" -ge "$((RETRIES+1))" ]]; then
      die "scp falló tras $attempt intentos: $remote_path"
    fi
    sleep 2
  done
}

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  for rf in "${remote_files[@]}"; do
    scp_one "$rf" "$OUTDIR"
  done
else
  mapfile -t downloaded < <(find "$OUTDIR" -maxdepth 1 -type f -name "${PREFIX}${DATE_STR}*.tar.xz" -print | sort) || true
  [[ "${#downloaded[@]}" -gt 0 ]] || die "Con --skip-download, no veo archivos locales en $OUTDIR con pattern ${PREFIX}${DATE_STR}*.tar.xz"
fi

# -----------------------
# Extract (con chequeo básico de path traversal)
# -----------------------
tar_safe_check() {
  local f="$1"
  tar -tf "$f" | awk '
    ($0 ~ /^\//) || ($0 ~ /(^|\/)\.\.(\/|$)/) { bad=1 }
    END { exit(bad?1:0) }
  '
}

if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
  for f in "${downloaded[@]}"; do
    [[ -f "$f" ]] || die "No existe para extraer: $f"
    tar_safe_check "$f" || die "Tar inseguro (rutas absolutas o ..): $f"
    run_step "Extrayendo $(basename "$f")" tar -xf "$f" -C "$OUTDIR" --no-same-owner --no-same-permissions
    if [[ "$KEEP_ARCHIVES" -eq 0 ]]; then
      run_step "Borrando archive $(basename "$f")" rm -f -- "$f"
    fi
  done
else
  echo "[$(ts)] Extracción deshabilitada (--skip-extract)."
fi

# -----------------------
# Python steps
# -----------------------
# -----------------------
# Python steps
# -----------------------
if [[ "$SKIP_PYTHON" -eq 0 ]]; then
  CONCAT_NAME="archivo_concatenado.csv"

  (
    cd "$OUTDIR"

    run_step "Ejecutando traduccionMuTe.py en *_list.csv (carpeta=.)" \
      translate_list_csvs_in_cwd

    run_step "Borrando CSV originales (*_list.csv)" \
      delete_list_csvs_in_cwd

    run_step "Ejecutando unircsv.py (carpeta=.)" \
      "$PYTHON_BIN" "$SCRIPTS_DIR/unircsv.py" -o "$CONCAT_NAME" .

    [[ -s "$CONCAT_NAME" ]] || die "No se generó $OUTDIR/$CONCAT_NAME (revisa salida de unircsv.py)"

    run_step "Ordenando temporalmente $CONCAT_NAME por columna time" \
      sort_csv_by_time_in_place "$CONCAT_NAME"

    run_step "Ejecutando filtro_coincidencias.py (input=$CONCAT_NAME)" \
      "$PYTHON_BIN" "$SCRIPTS_DIR/filtro_coincidencias.py" "$CONCAT_NAME"
  )
else
  echo "[$(ts)] Python deshabilitado (--skip-python)."
fi


echo
echo "[$(ts)] ✅ Pipeline completo. Tiempo total: $(fmt_time "$SECONDS")"
echo "[$(ts)] Outdir: $(cd "$OUTDIR" && pwd)"
