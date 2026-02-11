# Imagen base ligera con Python
FROM python:3.11-slim

# Evitar preguntas interactivas de apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema:
# - Tesseract + idiomas (spa, eng)
# - poppler-utils para pdf2image
# - libgl1 para OpenCV
# - libglib2.0-0 (suele hacer falta también con cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY . .

# Por defecto usaremos el puerto 8501 de Streamlit
EXPOSE 8501

# Variable de entorno opcional (evita que Streamlit reclame el navegador)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Comando de arranque:
# Lanza Streamlit con tu script, escuchando en 0.0.0.0 para que el host pueda acceder.
ENTRYPOINT ["streamlit", "run", "panel_tool_name_edad.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
