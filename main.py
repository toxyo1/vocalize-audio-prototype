"""
Protótipo de captura contínua de áudio e detecção de palavra-chave.

Dependências:
    pip install vosk sounddevice

Modelo: vosk-model-small-pt-0.3
"""

import json
import queue
import sys

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

KEYWORD = "vocalize"
SAMPLE_RATE = 44100       # kz
BLOCK_SIZE = 4000
INPUT_DEVICE = 8       # dispositivo de áudio
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "model"

# True  -> o reconhecedor só procura a palavra-chave
# False -> transcreve o áudio normalmente
USE_GRAMMAR = False

# True -> imprime o que o vosk reconheceu
DEBUG = True


class MicrophoneSource:
    # captura contínua do microfone

    def __init__(self, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=INPUT_DEVICE):
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._stream = sd.RawInputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[aviso de áudio] {status}", file=sys.stderr)
        self._queue.put(bytes(indata))

    def chunks(self):
        with self._stream:
            while True:
                yield self._queue.get()


class KeywordSpotter:
    # recebe blocos de áudio e diz se a palavra-chave apareceu

    def __init__(self, model_path, keyword=KEYWORD, samplerate=SAMPLE_RATE):
        self.keyword = keyword
        model = Model(model_path)
        if USE_GRAMMAR:
            grammar = json.dumps([keyword, "[unk]"])
            self.recognizer = KaldiRecognizer(model, samplerate, grammar)
        else:
            self.recognizer = KaldiRecognizer(model, samplerate)

    def process(self, chunk: bytes) -> bool:
        if self.recognizer.AcceptWaveform(chunk):
            text = json.loads(self.recognizer.Result()).get("text", "")

            if DEBUG and text:
                print(f"[reconhecido] {text!r}")

            if self.keyword in text.lower().split():
                self.recognizer.Reset()
                return True

        return False


def main():
    SetLogLevel(-1)
    spotter = KeywordSpotter(MODEL_PATH)
    source = MicrophoneSource()

    print(f"Escutando... diga '{KEYWORD.upper()}' (Ctrl+C para sair)")
    for chunk in source.chunks():
        if spotter.process(chunk):
            print(f"Palavra-chave {KEYWORD.upper()} detectada!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEncerrado.")