#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

# tqdm opcional (si no está instalado, degradamos a prints simples)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def read_first_line_len(path: Path) -> int:
    """Devuelve el tamaño (en bytes) de la primera línea (header) incluyendo el salto de línea si existe."""
    with path.open("rb") as f:
        line = f.readline()
    return len(line)


def skip_first_line(f) -> int:
    """Avanza el file pointer hasta después de la primera línea y devuelve cuántos bytes se saltó."""
    start = f.tell()
    f.readline()  # lee hasta el primer \n (incluye \n si existe)
    return f.tell() - start


def copy_stream(src_f, dst_f, bufsize: int, pbar=None) -> Optional[int]:
    """Copia desde src_f a dst_f por bloques. Devuelve el último byte escrito (o None si no escribió nada)."""
    last_byte = None
    while True:
        chunk = src_f.read(bufsize)
        if not chunk:
            break
        dst_f.write(chunk)
        last_byte = chunk[-1]
        if pbar is not None:
            pbar.update(len(chunk))
    return last_byte


def ensure_trailing_newline(dst_f, last_byte: Optional[int], pbar=None) -> Optional[int]:
    """Asegura que el output termine en newline antes de pegar el siguiente archivo (evita pegado '...123ABC...')."""
    if last_byte is None:
        return last_byte
    if last_byte != 0x0A:  # b'\n'
        dst_f.write(b"\n")
        if pbar is not None:
            pbar.update(1)
        return 0x0A
    return last_byte


def list_csvs(folder: Path, output_path: Path) -> list[Path]:
    files = sorted(folder.glob("*.csv"))
    # Evitar incluir el propio output si cae en el mismo patrón
    files = [p for p in files if p.resolve() != output_path.resolve()]
    return files


def main(
    folder: str,
    output: str = "archivo_concatenado.csv",
    verify_header: bool = False,
    bufsize_mb: int = 8,
) -> int:
    carpeta = Path(folder).expanduser().resolve()
    if not carpeta.is_dir():
        print(f"ERROR: la ruta no es una carpeta: {carpeta}")
        return 2

    ruta_salida = (carpeta / output).resolve()

    archivos = list_csvs(carpeta, ruta_salida)
    if not archivos:
        print("No se encontraron archivos CSV en la carpeta especificada.")
        return 0

    # Borrar salida previa si existe
    if ruta_salida.exists():
        ruta_salida.unlink()

    # Header base (para opcionalmente verificar)
    base_header = None
    if verify_header:
        with archivos[0].open("rb") as f:
            base_header = f.readline()

    # Calcular bytes totales “reales” (sin header repetido)
    total_bytes = 0
    for i, p in enumerate(archivos):
        size = p.stat().st_size
        if i == 0:
            total_bytes += size
        else:
            total_bytes += max(0, size - read_first_line_len(p))

    bufsize = max(1024 * 1024, bufsize_mb * 1024 * 1024)

    print(f"Se encontraron {len(archivos)} archivos CSV.")
    print(f"Escribiendo salida en: {ruta_salida}")

    use_tqdm = tqdm is not None and total_bytes > 0

    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            desc="Concatenando",
        )

    last_byte_written: Optional[int] = None

    try:
        with ruta_salida.open("wb") as out_f:
            for idx, path in enumerate(archivos, start=1):
                # Progreso por archivos (además del de bytes)
                if not use_tqdm:
                    print(f"[{idx}/{len(archivos)}] {path.name}")

                with path.open("rb") as in_f:
                    if idx == 1:
                        # Copia completa del primer archivo (incluye header)
                        last_byte_written = copy_stream(in_f, out_f, bufsize, pbar=pbar)
                    else:
                        # Opcional: verificar header igual
                        if verify_header:
                            header = in_f.readline()
                            if header != base_header:
                                raise ValueError(
                                    f"Header distinto en {path.name}.\n"
                                    f"Esperado: {base_header!r}\n"
                                    f"Encontrado: {header!r}"
                                )
                        else:
                            # Saltar header sin interpretar CSV
                            skip_first_line(in_f)

                        # Asegurar newline entre archivos si hace falta
                        last_byte_written = ensure_trailing_newline(out_f, last_byte_written, pbar=pbar)

                        # Copiar el resto
                        last_byte_written = copy_stream(in_f, out_f, bufsize, pbar=pbar)

        if pbar is not None:
            pbar.close()

        print("Concatenación finalizada.")
        return 0

    except Exception as e:
        if pbar is not None:
            pbar.close()
        print(f"ERROR: {e}")
        # Si quedó un output parcial, lo borramos para no dejar basura
        try:
            if ruta_salida.exists():
                ruta_salida.unlink()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatena CSVs en una carpeta de forma ultra eficiente (streaming) y con progreso."
    )
    parser.add_argument("carpeta", type=str, help="Ruta a la carpeta con archivos CSV")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="archivo_concatenado.csv",
        help="Nombre del archivo de salida (por defecto: archivo_concatenado.csv)",
    )
    parser.add_argument(
        "--verify-header",
        action="store_true",
        help="Falla si algún archivo tiene un header diferente al primero.",
    )
    parser.add_argument(
        "--bufsize-mb",
        type=int,
        default=8,
        help="Tamaño del buffer de copia en MB (más grande suele ser más rápido en disco).",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.carpeta, args.output, args.verify_header, args.bufsize_mb))
