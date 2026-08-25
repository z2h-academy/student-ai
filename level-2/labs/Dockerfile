# level-2/labs/Dockerfile
# Imagen de produccion para la API librarian.
#
# Multi-stage: primero se instalan las dependencias (con cache),
# luego se copia la app al runtime final.

# --- Stage 1: dependencias ---
FROM python:3.12-slim AS deps

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- Stage 2: runtime ---
FROM python:3.12-slim

WORKDIR /code

# Copia las dependencias instaladas en el stage anterior
# (site-packages + binarios como el ejecutable `uvicorn`)
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copia el codigo de la app
COPY ./labs /code/labs
COPY ./data /code/data

# Expone el puerto del servidor
EXPOSE 80

# Comando de inicio: un solo worker (el orquestador escala contenedores)
CMD ["uvicorn", "labs.api_main:app", "--host", "0.0.0.0", "--port", "80"]
