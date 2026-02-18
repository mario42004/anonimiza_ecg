# anonimiza_ecg

Herramienta para procesar PDFs clínicos de ECG: recorta el panel del trazado, asigna un identificador anónimo y extrae por OCR el encabezado con sus campos. Incluye una interfaz en Streamlit para procesamiento por lote.

## Qué hace
- Detecta el panel ECG (zona rosa) y lo recorta.
- Genera un ID anónimo y guarda los paneles con nombre `panel_<id>.png`.
- Extrae el encabezado del PDF/imagen y aplica OCR.
- Genera un CSV con metadatos (`rapidaim_id`, `panel_id`, `patient_id`, `panel_filename`, `source_file`, `raw_data`, `paper_speed`, `amplitude`, `freq_prefilter`).
- Exporta archivos auxiliares por documento (máscaras, header, TSV, JSON).

## Requisitos
- Docker y Docker Compose

## Ejecutar con Docker Compose
```bash
docker compose up --build
```
La app quedará disponible en `http://localhost:8501`.

## Uso
1. Sube PDFs/imagenes desde la interfaz o apunta a una carpeta local (por defecto `pdf_inputs/`).
2. Ajusta parámetros si hace falta (HSV, padding, OCR).
3. Ejecuta el procesamiento en lote.

## Salidas
Por defecto se guardan en `ecg_outputs/`:
- `panels/` con los recortes anonimizados.
- `ecg_panels_metadata.csv` con campos OCR y trazabilidad.
- Archivos por documento: `*_pink_mask.png`, `*_header_text.png`, `*_header_text.txt`, `*_header_ocr.tsv`, `*_meta.json`.

## Notas
- Los archivos `.png`, `.pdf` y `.csv` y los directorios `pdf_inputs/` y `ecg_outputs/` están ignorados en git.
