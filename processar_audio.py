"""
Processamento do áudio (etapa 2): acha o comando VOCALIZE e recorta a vocalização.

Fluxo:
  1. divide o áudio nos silêncios em uma lista de trechos
  2. reconhece só as palavras de cada trecho (comando e rótulos), sem transcrever tudo
  3. ao achar VOCALIZE:
        vocalização -> o áudio imediatamente ANTES do comando (recortado, nunca transcrito)
        rótulo      -> a palavra de rótulo logo DEPOIS do comando
  4. salva cada vocalização em um .wav para você ouvir e conferir

Dependências:
    pip install vosk pydub
Modelo em português (baixar e descompactar como "model/"):
    https://alphacephei.com/vosk/models  ->  vosk-model-small-pt-0.3

Uso:
    python processar_audio.py uploads/ARQUIVO.wav [pasta_do_modelo]
"""

import json
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from vosk import KaldiRecognizer, Model, SetLogLevel

KEYWORD = "vocalize"
LABELS = ["raiva", "alegria", "tristeza"]   # PROVISÓRIO: trocar pelos rótulos definidos

# --- Divisão por silêncio (ajustar olhando a saída de depuração) ---
MIN_SILENCE_MS = 400          # pausa mínima para separar dois trechos
SILENCE_THRESH_DBFS = None    # None = automático (média do áudio - 16 dB); ou ex.: -45
PADDING_MS = 150              # folga colocada em volta de cada trecho

# --- Recorte da vocalização ---
MAX_VOCALIZATION_MS = 5000    # duração máxima do recorte antes do comando
MIN_PRE_MS = 300              # se há mais que isso de áudio antes do comando no MESMO
                              # trecho (sem pausa), a vocalização sai de dentro dele

# --- Reconhecimento ---
# False: vocabulário completo, procura as palavras na lista (evita que um som qualquer,
#        como "aoaoao", seja forçado a virar "vocalize")
# True:  restringe o reconhecedor a KEYWORD + LABELS (mais sensível, mais falsos positivos)
USE_GRAMMAR = False
MIN_CONF = 0.5                # confiança mínima de uma palavra para ser aceita
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "model"
OUTPUT_ROOT = Path("saida")


def segmentar(audio: AudioSegment) -> list[tuple[int, int]]:
    """Divide o áudio nos silêncios. Devolve [(início_ms, fim_ms), ...]."""
    limiar = SILENCE_THRESH_DBFS if SILENCE_THRESH_DBFS is not None else audio.dBFS - 16
    faixas = detect_nonsilent(
        audio, min_silence_len=MIN_SILENCE_MS, silence_thresh=limiar, seek_step=10
    )
    trechos = []
    ultimo_fim = 0
    for inicio, fim in faixas:
        inicio = max(ultimo_fim, inicio - PADDING_MS)
        fim = min(len(audio), fim + PADDING_MS)
        trechos.append((inicio, fim))
        ultimo_fim = fim
    return trechos


def reconhecer(model: Model, trecho: AudioSegment) -> list[dict]:
    """Palavras reconhecidas em um trecho: [{word, start_ms, end_ms, conf}, ...]
    (tempos relativos ao início do trecho)."""
    trecho = trecho.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    if USE_GRAMMAR:
        gramatica = json.dumps([KEYWORD, *LABELS, "[unk]"])
        rec = KaldiRecognizer(model, 16000, gramatica)
    else:
        rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    dados = trecho.raw_data
    for i in range(0, len(dados), 8000):
        rec.AcceptWaveform(dados[i:i + 8000])
    resultado = json.loads(rec.FinalResult())

    return [
        {
            "word": w["word"].lower(),
            "start_ms": int(w["start"] * 1000),
            "end_ms": int(w["end"] * 1000),
            "conf": w.get("conf", 1.0),
        }
        for w in resultado.get("result", [])
        if w.get("conf", 1.0) >= MIN_CONF
    ]


def localizar(trechos: list[dict]) -> list[dict]:
    """Recebe [{inicio, fim, palavras}, ...] e devolve uma ocorrência por comando:
    {comando_ms, vocalizacao: (ini, fim) | None, rotulo: str | None}"""
    ocorrencias = []

    for i, trecho in enumerate(trechos):
        palavras = trecho["palavras"]
        idx = next((k for k, w in enumerate(palavras) if w["word"] == KEYWORD), None)
        if idx is None:
            continue

        comando_ms = trecho["inicio"] + palavras[idx]["start_ms"]

        # Vocalização: dentro do mesmo trecho (sem pausa) ou no trecho anterior
        if palavras[idx]["start_ms"] >= MIN_PRE_MS:
            fim = comando_ms
            inicio = max(trecho["inicio"], fim - MAX_VOCALIZATION_MS)
            vocalizacao = (inicio, fim)
        elif i > 0:
            anterior = trechos[i - 1]
            fim = anterior["fim"]
            inicio = max(anterior["inicio"], fim - MAX_VOCALIZATION_MS)
            vocalizacao = (inicio, fim)
        else:
            vocalizacao = None

        # Rótulo: depois do comando no mesmo trecho ou no trecho seguinte
        candidatos = [w["word"] for w in palavras[idx + 1:]]
        if i + 1 < len(trechos):
            candidatos += [w["word"] for w in trechos[i + 1]["palavras"]]
        rotulo = next((p for p in candidatos if p in LABELS), None)

        ocorrencias.append(
            {"comando_ms": comando_ms, "vocalizacao": vocalizacao, "rotulo": rotulo}
        )

    return ocorrencias


def formatar(ms: int) -> str:
    return f"{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python processar_audio.py ARQUIVO.wav [pasta_do_modelo]")

    caminho = Path(sys.argv[1])
    SetLogLevel(-1)
    model = Model(MODEL_PATH)
    audio = AudioSegment.from_file(caminho)
    print(f"Áudio: {caminho.name} ({formatar(len(audio))})")

    faixas = segmentar(audio)
    print(f"{len(faixas)} trechos encontrados\n")

    trechos = []
    for n, (inicio, fim) in enumerate(faixas):
        palavras = reconhecer(model, audio[inicio:fim])
        trechos.append({"inicio": inicio, "fim": fim, "palavras": palavras})
        texto = " ".join(f"{w['word']}({w['conf']:.2f})" for w in palavras) or "—"
        print(f"[{n:02d}] {formatar(inicio)} → {formatar(fim)}  {texto}")

    ocorrencias = localizar(trechos)
    print(f"\n{len(ocorrencias)} comando(s) {KEYWORD.upper()} encontrado(s)")

    pasta = OUTPUT_ROOT / caminho.stem
    pasta.mkdir(parents=True, exist_ok=True)
    for n, oc in enumerate(ocorrencias, start=1):
        arquivo = "(sem vocalização antes do comando)"
        if oc["vocalizacao"]:
            ini, fim = oc["vocalizacao"]
            destino = pasta / f"vocalizacao_{n:02d}.wav"
            audio[ini:fim].export(destino, format="wav")
            arquivo = f"{destino}  ({formatar(ini)} → {formatar(fim)})"
        print(f"  #{n} comando em {formatar(oc['comando_ms'])} | "
              f"rótulo: {oc['rotulo'] or 'não identificado'} | vocalização: {arquivo}")


if __name__ == "__main__":
    main()