"""
Cliente de gravação (etapas 1, 2 e 3):
  1. grava continuamente pelo microfone
  2. permite iniciar e parar a gravação (ENTER)
  3. ao parar, envia o arquivo de áudio inteiro para a API

Rodar (com a API já no ar):
    python recorder.py
"""

import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf

API_URL = "http://localhost:8000/recordings"
SAMPLE_RATE = None
CHANNELS = 1
INPUT_DEVICE = 8 
OUTPUT_DIR = Path("recordings")


class Recorder:
    """Grava o microfone direto em um arquivo WAV, bloco a bloco.

    Como o áudio vai para o disco durante a gravação, no fim já existe o
    arquivo completo e uma falha no meio não perde tudo o que foi gravado.
    """

    def __init__(self, samplerate=SAMPLE_RATE, channels=CHANNELS, device=INPUT_DEVICE):
        if samplerate is None:
            samplerate = int(sd.query_devices(device, "input")["default_samplerate"])
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._queue: "queue.Queue" = queue.Queue()
        self._stop_event = threading.Event()
        self._stream = None
        self._thread = None
        self._file = None
        self.path = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[aviso de áudio] {status}", file=sys.stderr)
        self._queue.put(indata.copy())

    def _writer(self):
        # Esvazia a fila até receber o sinal de parada e não restar nada nela
        while not (self._stop_event.is_set() and self._queue.empty()):
            try:
                block = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._file.write(block)

    def start(self) -> Path:
        OUTPUT_DIR.mkdir(exist_ok=True)
        name = datetime.now().strftime("sessao_%Y%m%d_%H%M%S.wav")
        self.path = OUTPUT_DIR / name

        self._stop_event.clear()
        self._file = sf.SoundFile(
            self.path, mode="w", samplerate=self.samplerate,
            channels=self.channels, subtype="PCM_16",
        )
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=self.channels,
            device=self.device, dtype="int16", callback=self._callback,
        )
        self._stream.start()
        return self.path

    def stop(self) -> Path:
        self._stream.stop()
        self._stream.close()
        self._stop_event.set()
        self._thread.join()
        self._file.close()
        return self.path


def send_to_api(path: Path) -> bool:
    """Envia o áudio inteiro. Se falhar, o arquivo continua salvo localmente."""
    try:
        with path.open("rb") as f:
            response = requests.post(
                API_URL, files={"file": (path.name, f, "audio/wav")}, timeout=300
            )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Falha ao enviar: {error}")
        print(f"O áudio continua salvo em: {path}")
        return False

    print(f"Enviado com sucesso: {response.json()}")
    return True


def main():
    recorder = Recorder()

    input("Pressione ENTER para INICIAR a gravação...")
    path = recorder.start()
    print(f"Gravando em {path} ... pressione ENTER para PARAR.")

    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Gravação encerrada ({size_mb:.2f} MB). Enviando para a API...")
    send_to_api(path)


if __name__ == "__main__":
    main()