#developed by Mario Jojo Acosta
#date: 2025-12-07
#version: 1.0           
#license: MIT


import io
import csv
import tempfile
from pathlib import Path
import zipfile
import time
import re
import getpass  # para detectar /media/<usuario>
import subprocess

import streamlit as st
from extract_all import run_pipeline

st.set_page_config(page_title="ECG Extractor", layout="wide")
st.title("🫀 Extracción automática de panel ECG y texto de cabecera (batch)")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_zip_from_map(files_map):
    """
    files_map: dict[Path, str]  -> {ruta_real: nombre_dentro_del_zip}
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in files_map.items():
            zf.write(path, arcname=arcname)
    buf.seek(0)
    return buf


def generate_unique_id() -> str:
    """Genera un ID único usando timestamp Unix (epoch) en nanosegundos."""
    return str(time.time_ns())


def pdf_extract_text(pdf_path: Path) -> str:
    """
    Extrae texto embebido de un PDF usando pdftotext (poppler-utils).
    Devuelve string vacío si falla.
    """
    try:
        res = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout or ""
    except Exception:
        return ""


def extract_footer_fields_from_text(text: str) -> dict:
    """
    Extrae velocidad de papel, amplitud y frecuencia de muestreo desde texto.
    Soporta patrones tipo: 25mm/s, 10mm/mV, 150Hz.
    """
    def _find(pattern: str) -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else ""

    paper_speed = _find(r"(\d+(?:\.\d+)?)\s*mm\s*/\s*s")
    amplitude = _find(r"(\d+(?:\.\d+)?)\s*mm\s*/\s*mV")
    freq_prefilter = _find(r"(\d+(?:\.\d+)?)\s*Hz")

    return {
        "paper_speed": paper_speed,
        "amplitude": amplitude,
        "freq_prefilter": freq_prefilter,
    }


def parse_header_to_fields(header_path: Path) -> dict:
    """
    Lee el archivo *_header_text.txt y devuelve solo:
      - raw_data (todo el texto del header aplanado)
    """
    base_fields = {
        "raw_data": "",
    }

    if header_path is None or not header_path.exists():
        return base_fields.copy()

    raw_text = header_path.read_text(encoding="utf-8", errors="ignore")

    # Aplanamos el texto en una sola línea (para raw_data), limpiando líneas vacías
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    flat_text = " | ".join(lines)

    fields = {
        "raw_data": flat_text,
    }

    return fields


def update_metadata_csv(csv_path: Path, new_row: dict):
    """
    Actualiza (o crea) un CSV con metadatos:
      - Lee contenido previo si existe.
      - Añade la nueva fila.
      - Reescribe el CSV con la unión de todas las columnas.
    """
    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    existing_rows.append(new_row)

    # Unimos todos los campos existentes
    fieldnames = set()
    for row in existing_rows:
        fieldnames.update(row.keys())

    preferred_order = [
        "rapidaim_id",
        "patient_id",
        "panel_id",
        "panel_filename",
        "source_file",
        "paper_speed",
        "amplitude",
        "freq_prefilter",
        "raw_data",
    ]
    ordered_present = [k for k in preferred_order if k in fieldnames]
    remaining = sorted([k for k in fieldnames if k not in preferred_order])
    fieldnames = ordered_present + remaining

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def load_existing_csv_state(csv_path: Path):
    """
    Lee el CSV existente y devuelve:
      - max_rapidaim_id: mayor rapidaim_id encontrado (0 si no hay)
      - source_files: conjunto de source_file ya registrados
    """
    max_rapidaim_id = 0
    source_files = set()

    if not csv_path.exists():
        return max_rapidaim_id, source_files

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = (row.get("source_file") or "").strip()
            if src:
                source_files.add(src)

            rid = (row.get("rapidaim_id") or "").strip()
            if rid.isdigit():
                max_rapidaim_id = max(max_rapidaim_id, int(rid))

    return max_rapidaim_id, source_files


def extract_patient_id_from_filename(original_name: str) -> str:
    """
    Extrae patient_id como el texto antes del primer underscore en el nombre.
    Ejemplo: 12345_abc_def.pdf -> 12345
    """
    filename = Path(original_name).name
    stem = Path(filename).stem
    return stem.split("_", 1)[0].strip()


def pdf_to_image(pdf_path: Path) -> Path:
    """
    Convierte la PRIMERA página de un PDF a una imagen PNG temporal
    y devuelve la ruta de esa imagen.
    """
    from pdf2image import convert_from_path

    pages = convert_from_path(str(pdf_path), dpi=300)
    if not pages:
        raise ValueError(f"No se pudieron convertir páginas del PDF: {pdf_path}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pages[0].save(tmp.name, "PNG")
    return Path(tmp.name)


def process_single_document(
    src_path: Path,
    original_name: str,
    out_root_dir: Path,
    panels_dir: Path,
    hsv_params: dict,
    pad_panel: int,
    header_min_height: int,
    lang: str,
    csv_path: Path,
    rapidaim_id: int,
    patient_id: str,
):
    """
    Procesa un único documento (PDF o imagen):
      - Si es PDF, convierte la primera página a imagen.
      - Crea una subcarpeta específica para ese documento.
      - Ejecuta run_pipeline escribiendo en esa subcarpeta.
      - Localiza el ECG panel y el header_text (por patrón *_ecg_panel.png, *_header_text.txt).
      - Mueve el ECG panel a la carpeta global 'panels' con un ID único (nombre estable).
      - Actualiza el CSV de metadatos con los campos del header.
    """
    # Subcarpeta con nombre basado en el nombre original (para que sea legible)
    doc_out_dir = out_root_dir / Path(original_name).stem
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    # Si el origen es PDF, extraer texto embebido y convertir a imagen antes de llamar a run_pipeline
    src_for_pipeline = src_path
    footer_fields = {"paper_speed": "", "amplitude": "", "freq_prefilter": ""}
    if src_path.suffix.lower() == ".pdf":
        try:
            embedded_text = pdf_extract_text(src_path)
            if embedded_text:
                footer_fields = extract_footer_fields_from_text(embedded_text)
            src_for_pipeline = pdf_to_image(src_path)
        except Exception as e:
            st.error(f"Error convirtiendo PDF a imagen para {original_name}: {e}")
            return None

    # Ejecutar pipeline: escribirá sus outputs en doc_out_dir
    run_pipeline(
        image_path=str(src_for_pipeline),
        out_dir=str(doc_out_dir),
        hsv_tune=hsv_params,
        pad_panel=pad_panel,
        header_min_height=header_min_height,
        lang=lang,
    )

    # Buscar panel por patrón: *_ecg_panel.png
    panel_candidates = sorted(doc_out_dir.glob("*_ecg_panel.png"))
    if not panel_candidates:
        st.warning(f"No se encontró panel ECG para {original_name}")
        return None
    ecg_panel_path = panel_candidates[0]  # si hay varios, cogemos el primero

    # Buscar header_text por patrón: *_header_text.txt (si existe)
    header_txt_candidates = sorted(doc_out_dir.glob("*_header_text.txt"))
    header_txt_path = header_txt_candidates[0] if header_txt_candidates else None

    # Asegurar carpeta global de panels
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Generar ID único y mover el panel a la carpeta global
    unique_id_raw = generate_unique_id()
    panel_id = f"_{unique_id_raw}_{rapidaim_id}"
    new_panel_name = f"panel_{unique_id_raw}_{rapidaim_id}.png"
    new_panel_path = panels_dir / new_panel_name
    ecg_panel_path.rename(new_panel_path)

    # Parsear header — devuelve raw_data
    header_fields = parse_header_to_fields(header_txt_path)

    # Fila de metadatos para CSV
    row = {
        "rapidaim_id": rapidaim_id,
        "panel_id": panel_id,
        "patient_id": patient_id,
        "panel_filename": new_panel_name,
        "source_file": original_name,
    }
    row.update(header_fields)
    row.update(footer_fields)

    # Redundancia defensiva: asegurar claves normalizadas
    for k in ("raw_data", "paper_speed", "amplitude", "freq_prefilter"):
        row.setdefault(k, "")

    # Actualizar CSV
    update_metadata_csv(csv_path, row)

    return {
        "panel_id": panel_id,
        "panel_path": new_panel_path,
        "source_file": original_name,
        "header_fields": header_fields,
    }


def list_external_mounts():
    """
    Devuelve una lista de rutas de posibles dispositivos externos en /media/<usuario>.
    """
    user = getpass.getuser()
    media_root = Path("/media") / user
    if not media_root.exists():
        return []
    return [p for p in media_root.iterdir() if p.is_dir()]


# ── UI de configuración ───────────────────────────────────────────────────────

st.sidebar.header("⚙️ Parámetros de procesamiento")
out_dir_str = st.sidebar.text_input("Carpeta raíz de salida (relativa)", value="ecg_outputs")

st.sidebar.caption(
    "Puedes escribir una ruta absoluta (ej. /media/USUARIO/USB/ecg_outputs) "
    "o usar la detección automática de USB más abajo."
)

# Detección de USB externa
usb_dirs = list_external_mounts()
use_usb = False
selected_usb = None

if usb_dirs:
    st.sidebar.subheader("💾 Salida en USB externa (opcional)")
    use_usb = st.sidebar.checkbox("Usar una USB detectada en /media", value=False)
    if use_usb:
        selected_usb = st.sidebar.selectbox(
            "Selecciona dispositivo USB",
            options=usb_dirs,
            format_func=lambda p: p.name,
        )
        st.sidebar.write(f"Ruta seleccionada: `{selected_usb}`")
else:
    st.sidebar.caption("No se han detectado dispositivos en /media/<usuario>. Inserta una USB si quieres usarla.")

pad_panel = st.sidebar.slider("Padding del panel ECG", 0, 20, 6)
header_min_height = st.sidebar.slider("Altura mínima del header", 10, 200, 40)
lang = st.sidebar.text_input("Idiomas Tesseract", value="spa+eng")

# HSV tuning
st.sidebar.subheader("🎨 Parámetros HSV (detección de zona rosa)")
h_low1 = st.sidebar.slider("h_low1", 0, 179, 0)
s_low1 = st.sidebar.slider("s_low1", 0, 255, 20)
v_low1 = st.sidebar.slider("v_low1", 0, 255, 140)
h_up1 = st.sidebar.slider("h_up1", 0, 179, 12)
s_up1 = st.sidebar.slider("s_up1", 0, 255, 200)
v_up1 = st.sidebar.slider("v_up1", 0, 255, 255)
h_low2 = st.sidebar.slider("h_low2", 0, 179, 165)
s_low2 = st.sidebar.slider("s_low2", 0, 255, 20)
v_low2 = st.sidebar.slider("v_low2", 0, 255, 140)
h_up2 = st.sidebar.slider("h_up2", 0, 179, 179)
s_up2 = st.sidebar.slider("s_up2", 0, 255, 200)
v_up2 = st.sidebar.slider("v_up2", 0, 255, 255)

hsv_params = dict(
    h_low1=h_low1, s_low1=s_low1, v_low1=v_low1,
    h_up1=h_up1,   s_up1=s_up1,   v_up1=v_up1,
    h_low2=h_low2, s_low2=s_low2, v_low2=v_low2,
    h_up2=h_up2,   s_up2=s_up2,   v_up2=v_up2,
)

# Origen de datos: subidos o carpeta local
st.sidebar.subheader("📂 Origen de documentos")
mode = st.sidebar.radio("Selecciona origen", ["Subir archivos", "Carpeta de PDFs"])

docs_to_process = []

if mode == "Subir archivos":
    uploaded_files = st.file_uploader(
        "📄 Sube uno o varios PDF/imagenes de ECG",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.write(f"Archivos subidos: {[f.name for f in uploaded_files]}")
        for uf in uploaded_files:
            suffix = Path(uf.name).suffix.lower() or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = Path(tmp.name)
            # guardamos (ruta temporal, nombre original)
            docs_to_process.append((tmp_path, uf.name))
else:
    pdf_folder_str = st.text_input("Ruta de la carpeta con PDFs", value="pdf_inputs")
    pdf_folder = Path(pdf_folder_str)
    if pdf_folder.exists():
        pdf_paths = sorted(pdf_folder.glob("*.pdf"))
        if pdf_paths:
            st.write(f"Se encontraron {len(pdf_paths)} PDFs en `{pdf_folder}`")
            docs_to_process = [(p, p.name) for p in pdf_paths]
        else:
            st.info("No se encontraron PDFs en la carpeta.")
    else:
        st.warning("La carpeta indicada no existe.")

# ── Ejecución batch ──────────────────────────────────────────────────────────

# Determinar carpeta raíz de salida final
if use_usb and selected_usb is not None:
    out_root_dir = selected_usb / out_dir_str
else:
    out_root_dir = Path(out_dir_str)

out_root_dir.mkdir(parents=True, exist_ok=True)

panels_dir = out_root_dir / "panels"
csv_path = out_root_dir / "ecg_panels_metadata.csv"

if docs_to_process:
    if st.button("🚀 Ejecutar extracción para TODOS los documentos"):
        results_summary = []
        duplicate_source_files = []
        failed_files = []
        max_existing_rapidaim_id, known_source_files = load_existing_csv_state(csv_path)
        next_rapidaim_id = max_existing_rapidaim_id + 1

        with st.spinner("Procesando documentos..."):
            for idx, (doc_path, original_name) in enumerate(docs_to_process, start=1):
                st.write(f"Procesando ({idx}/{len(docs_to_process)}): {original_name}")
                if original_name in known_source_files:
                    duplicate_source_files.append(original_name)
                    st.warning(f"Duplicado detectado (source_file ya existe): {original_name}. Se omite del CSV.")
                    continue

                patient_id = extract_patient_id_from_filename(original_name)
                res = process_single_document(
                    src_path=doc_path,
                    original_name=original_name,
                    out_root_dir=out_root_dir,
                    panels_dir=panels_dir,
                    hsv_params=hsv_params,
                    pad_panel=pad_panel,
                    header_min_height=header_min_height,
                    lang=lang,
                    csv_path=csv_path,
                    rapidaim_id=next_rapidaim_id,
                    patient_id=patient_id,
                )
                if res is not None:
                    results_summary.append(res)
                    known_source_files.add(original_name)
                    next_rapidaim_id += 1
                else:
                    failed_files.append(original_name)

        st.success("✅ Procesamiento batch completado.")

        st.subheader("📋 Reporte de ejecución")
        st.write(f"Total recibidos: {len(docs_to_process)}")
        st.write(f"Agregados al CSV: {len(results_summary)}")
        st.write(f"Omitidos por source_file duplicado: {len(duplicate_source_files)}")
        st.write(f"Fallidos por error de procesamiento: {len(failed_files)}")

        if duplicate_source_files:
            st.error("Se detectaron source_file repetidos. No fueron añadidos al CSV.")
            st.write("Duplicados:")
            st.write(duplicate_source_files)
        else:
            st.success("No se detectaron source_file repetidos.")

        if results_summary:
            st.subheader("Ejemplos de paneles generados")
            cols = st.columns(3)
            for i, item in enumerate(results_summary[:3]):
                with cols[i]:
                    st.image(
                        str(item["panel_path"]),
                        caption=f"{item['panel_id']} ({item['source_file']})",
                        use_column_width=True,
                    )

        st.info(f"📊 Metadatos guardados/actualizados en: `{csv_path}`")
        st.info(f"📂 Paneles guardados en: `{panels_dir}`")

        # ZIP conjunto: panels + CSV
        files_map = {}

        if csv_path.exists():
            files_map[csv_path] = "ecg_panels_metadata.csv"

        panel_paths = list(panels_dir.glob("panel_*.png"))
        for p in panel_paths:
            files_map[p] = f"panels/{p.name}"

        if files_map:
            zip_buf = make_zip_from_map(files_map)
            st.download_button(
                "🧩 Descargar TODO (panels + CSV) en un ZIP",
                data=zip_buf.getvalue(),
                file_name="ecg_panels_full.zip",
                mime="application/zip",
            )
else:
    st.info("👆 Sube archivos o indica una carpeta de PDFs para comenzar el análisis.")
