"""
API de recepção/tratamento (etapa 1): recebe o áudio inteiro ao fim da gravação
e salva em disco.

Rodar:
    uvicorn api:app --reload --port 8000
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav"}

app = FastAPI(title="VocalizeAI - API de recepção de áudio")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recordings", status_code=201)
def receive_recording(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Envie um arquivo .wav")

    recording_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = UPLOAD_DIR / f"{timestamp}_{recording_id}{extension}"

    # Copia em blocos para não carregar o áudio inteiro na memória
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    size = destination.stat().st_size
    if size == 0:
        destination.unlink()
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    return {
        "id": recording_id,
        "filename": destination.name,
        "size_bytes": size,
    }