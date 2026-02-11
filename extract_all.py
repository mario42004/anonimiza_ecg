import streamlit as st
import cv2
import json
import pytesseract
import numpy as np
from pathlib import Path
from typing import Tuple
#python extract_all.py   --image ./imgs/testing.jpg   --out-dir ecg_outputs   --pad-panel 6   --lang spa+eng
def find_pink_mask_bgr(img_bgr: np.ndarray,
                       h_low1=0, s_low1=20, v_low1=140, h_up1=12, s_up1=200, v_up1=255,
                       h_low2=165, s_low2=20, v_low2=140, h_up2=179, s_up2=200, v_up2=255,
                       open_ksz=3, close_ksz=3, close_iter=2) -> np.ndarray:
    """Máscara binaria (0/255) de la zona rosa en HSV con dos rangos (rojo bajo y alto)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([h_low1, s_low1, v_low1], np.uint8)
    upper1 = np.array([h_up1,  s_up1,  v_up1 ], np.uint8)
    lower2 = np.array([h_low2, s_low2, v_low2], np.uint8)
    upper2 = np.array([h_up2,  s_up2,  v_up2 ], np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    if open_ksz > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksz, open_ksz))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se, iterations=1)
    if close_ksz > 1 and close_iter > 0:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksz, close_ksz))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se, iterations=close_iter)

    return mask

def largest_rect_from_mask(mask: np.ndarray) -> Tuple[int,int,int,int]:
    """BBox axis-aligned (x0,y0,x1,y1) del mayor contorno en la máscara (255=positivo)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0,0,0,0)
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return (x, y, x+w-1, y+h-1)

def clamp_bbox(x0,y0,x1,y1,w,h,pad=0):
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w-1, x1 + pad); y1 = min(h-1, y1 + pad)
    if x1 < x0: x0, x1 = 0, 0
    if y1 < y0: y0, y1 = 0, 0
    return x0,y0,x1,y1

def depink_for_ocr(img_bgr: np.ndarray, pink_mask: np.ndarray) -> np.ndarray:
    """Neutraliza el rosa (canal A en Lab) donde hay máscara rosa para mejorar contraste del texto."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A2 = A.copy()
    A2[pink_mask>0] = 128  # neutraliza rojizo
    out = cv2.cvtColor(cv2.merge([L, A2, B]), cv2.COLOR_LAB2BGR)
    return out

def ocr_image(img_bgr: np.ndarray, lang="spa+eng", psm=6, oem=3):
    """Ejecuta Tesseract y devuelve texto y TSV con boxes."""
    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(img_bgr, lang=lang, config=config)
    tsv  = pytesseract.image_to_data(img_bgr, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME)
    return text, tsv

def run_pipeline(image_path: str,
                 out_dir: str = "ecg_outputs",
                 hsv_tune: dict = None,
                 pad_panel: int = 6,
                 header_min_height: int = 40,
                 lang: str = "spa+eng"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = Path(image_path).stem

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H,W = img.shape[:2]

    # 1) máscara rosa (panel ECG)
    hsv_params = hsv_tune or {}
    pink_mask = find_pink_mask_bgr(img, **hsv_params)
    cv2.imwrite(f"{out_dir}/{base}_pink_mask.png", pink_mask)

    # 2) bbox del panel ECG
    x0,y0,x1,y1 = largest_rect_from_mask(pink_mask)
    x0,y0,x1,y1 = clamp_bbox(x0,y0,x1,y1,W,H,pad=pad_panel)

    # 3) recortes
    ecg_panel = img[y0:y1+1, x0:x1+1].copy()
    cv2.imwrite(f"{out_dir}/{base}_ecg_panel.png", ecg_panel)

    # header = todo lo que está arriba del panel (si hay espacio)
    header_top, header_bottom = 0, max(0, y0-1)
    header = img[header_top:header_bottom, :].copy() if header_bottom-header_top >= header_min_height else None
    if header is not None and header.size:
        cv2.imwrite(f"{out_dir}/{base}_header_text.png", header)

    # 4) OCR del header (mejorando contraste contra rosa residual)
    ocr_txt = ""
    if header is not None and header.size:
        # de-pink en toda la imagen + recorta misma zona (por si hay rosa tenue detrás del texto)
        depinked = depink_for_ocr(img, pink_mask)
        header_depink = depinked[header_top:header_bottom, :].copy()
        # pre-proceso leve para OCR
        gray = cv2.cvtColor(header_depink, cv2.COLOR_BGR2GRAY)
        # umbral adaptativo inverso (texto claro sobre fondo)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 35, 8)
        cv2.imwrite(f"{out_dir}/{base}_header_bin.png", thr)

        # OCR (usar imagen “depinked” original; Tesseract maneja mejor que el binario duro)
        text, tsv = ocr_image(header_depink, lang=lang, psm=6, oem=3)
        ocr_txt = text
        with open(f"{out_dir}/{base}_header_text.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # TSV con boxes
        try:
            tsv.to_csv(f"{out_dir}/{base}_header_ocr.tsv", sep="\t", index=False)
        except Exception:
            pass

    # 5) Exportar metadatos JSON útiles
    meta = {
        "image": str(Path(image_path).name),
        "size": [int(W), int(H)],
        "ecg_panel_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "header_bbox_xyxy": [0, 0, int(W-1), int(max(0, y0-1))],
        "ocr_text_preview": ocr_txt[:500]
    }
    with open(f"{out_dir}/{base}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Guardado en:", out_dir)
    print(" -", f"{base}_pink_mask.png")
    print(" -", f"{base}_ecg_panel.png")
    if header is not None:
        print(" -", f"{base}_header_text.png")
        print(" -", f"{base}_header_text.txt")
        print(" -", f"{base}_header_ocr.tsv")
    print(" -", f"{base}_meta.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extrae panel ECG y cabecera de texto; hace OCR del header.")
    ap.add_argument("--image", required=True, help="Ruta a la imagen (p.ej. ./imgs/testing_2.jpg)")
    ap.add_argument("--out-dir", default="ecg_outputs", help="Carpeta de salida")
    ap.add_argument("--pad-panel", type=int, default=6, help="Padding alrededor del panel ECG")
    ap.add_argument("--header-min-height", type=int, default=40, help="Altura mínima para considerar header")
    ap.add_argument("--lang", default="spa+eng", help="Idiomas Tesseract (ej: spa, eng, spa+eng)")
    # parámetros HSV opcionales
    ap.add_argument("--h-low1", type=int, default=0)
    ap.add_argument("--s-low1", type=int, default=20)
    ap.add_argument("--v-low1", type=int, default=140)
    ap.add_argument("--h-up1",  type=int, default=12)
    ap.add_argument("--s-up1",  type=int, default=200)
    ap.add_argument("--v-up1",  type=int, default=255)
    ap.add_argument("--h-low2", type=int, default=165)
    ap.add_argument("--s-low2", type=int, default=20)
    ap.add_argument("--v-low2", type=int, default=140)
    ap.add_argument("--h-up2",  type=int, default=179)
    ap.add_argument("--s-up2",  type=int, default=200)
    ap.add_argument("--v-up2",  type=int, default=255)
    args = ap.parse_args()

    hsv_params = dict(
        h_low1=args.h_low1, s_low1=args.s_low1, v_low1=args.v_low1,
        h_up1=args.h_up1,   s_up1=args.s_up1,   v_up1=args.v_up1,
        h_low2=args.h_low2, s_low2=args.s_low2, v_low2=args.v_low2,
        h_up2=args.h_up2,   s_up2=args.s_up2,   v_up2=args.v_up2,
    )

    run_pipeline(
        image_path=args.image,
        out_dir=args.out_dir,
        hsv_tune=hsv_params,
        pad_panel=args.pad_panel,
        header_min_height=args.header_min_height,
        lang=args.lang
    )
