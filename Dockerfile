# Usa Python oficial como base
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . .

# Actualiza pip y luego instala las dependencias
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8080 (requerido por Google Cloud Run)
EXPOSE 8080

# Variable de entorno para Flask
ENV PYTHONUNBUFFERED=True

# Comando para ejecutar la aplicación (ajústalo si tu archivo principal tiene otro nombre)
CMD ["python", "app.py"]
