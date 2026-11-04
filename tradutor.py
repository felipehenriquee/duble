"""Tradutor de voz ao vivo: fala em português no microfone, sai falado em inglês.

Microfone -> faster-whisper (traduz direto para inglês) -> Piper ou voz clonada -> alto-falante
ou microfone virtual (para usar no Meet, Discord, Zoom...).

Com --meet, também escuta o áudio do computador (a outra pessoa da chamada), transcreve o
inglês e mostra a tradução em português no terminal e em meet.txt.
"""

import argparse
import collections
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
TAXA_MIC = 16000
QUADRO_MS = 30
AMOSTRAS_QUADRO = TAXA_MIC * QUADRO_MS // 1000

SINK_VIRTUAL = "tradutor_saida"
MIC_VIRTUAL = "tradutor_mic"
MODELO_EN_PT = PASTA / "modelos" / "translate-en_pb-1_9"

# Frases que o Whisper costuma "inventar" em silêncio ou ruído (comparadas sem pontuação)
ALUCINACOES = {
    "thank you", "thanks for watching", "thank you for watching",
    "you", "bye", "bye bye", "thank you very much", "please subscribe",
}


def criar_microfone_virtual():
    """Cria um sink nulo e um microfone virtual que escuta esse sink. Devolve os ids dos módulos."""
    sink = subprocess.check_output([
        "pactl", "load-module", "module-null-sink",
        f"sink_name={SINK_VIRTUAL}",
        "sink_properties=device.description=Tradutor_(saida)",
    ], text=True).strip()
    mic = subprocess.check_output([
        "pactl", "load-module", "module-remap-source",
        f"master={SINK_VIRTUAL}.monitor",
        f"source_name={MIC_VIRTUAL}",
        "source_properties=device.description=Tradutor_(microfone_virtual)",
    ], text=True).strip()
    return [mic, sink]


def remover_modulos(ids):
    for i in ids:
        subprocess.run(["pactl", "unload-module", i], stderr=subprocess.DEVNULL)


def rms(quadro):
    return float(np.sqrt(np.mean(quadro.astype(np.float32) ** 2)))


class Gravador:
    """Lê uma entrada de áudio com parec e corta em frases usando o volume."""

    def __init__(self, dispositivo, silencio_ms, sensibilidade, trecho_ms, limiar=None):
        cmd = ["parec", "--format=s16le", f"--rate={TAXA_MIC}", "--channels=1",
               f"--latency-msec={QUADRO_MS}"]
        if dispositivo:
            cmd.append(f"--device={dispositivo}")
        # sem stderr: ao sair, o parec reclama de "pipe quebrado"; falhas reais viram EOFError em ler_quadro
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.quadros_silencio = silencio_ms // QUADRO_MS
        # depois de trecho_ms falando sem parar, uma pausa curta (respiração, vírgula) já fecha o trecho
        self.quadros_trecho = trecho_ms // QUADRO_MS
        self.quadros_pausa_curta = min(250 // QUADRO_MS, self.quadros_silencio)
        self.sensibilidade = sensibilidade
        self.limiar = limiar
        self.mudo = threading.Event()
        self.sons = collections.deque(maxlen=3000)  # momentos com som no último ~1,5 min

    def teve_som(self, inicio, fim):
        """Se esta entrada tinha som entre `inicio` e `fim` (usado para detectar eco da chamada)."""
        return any(inicio <= t <= fim for t in reversed(self.sons))

    def ler_quadro(self):
        dados = self.proc.stdout.read(AMOSTRAS_QUADRO * 2)
        if len(dados) < AMOSTRAS_QUADRO * 2:
            raise EOFError("o microfone parou de enviar áudio")
        return np.frombuffer(dados, dtype=np.int16)

    def calibrar(self, segundos=1.0):
        niveis = [rms(self.ler_quadro()) for _ in range(int(segundos * 1000 / QUADRO_MS))]
        ruido = float(np.median(niveis))
        self.limiar = max(ruido * self.sensibilidade, 250.0)
        return ruido

    def frases(self):
        """Devolve (áudio, momento em que a pessoa parou de falar)."""
        pre = []  # um pouco de áudio antes da fala, para não cortar o começo da palavra
        frase, falando, silencio, fortes = [], False, 0, 0
        max_quadros = 15000 // QUADRO_MS
        while True:
            q = self.ler_quadro()
            if self.mudo.is_set():
                # enquanto a voz traduzida toca, ignora a entrada para não traduzir a própria voz
                pre, frase, falando, silencio, fortes = [], [], False, 0, 0
                continue
            alto = rms(q) > self.limiar
            if alto:
                self.sons.append(time.time())
            if not falando:
                pre = (pre + [q])[-10:]
                fortes = fortes + 1 if alto else 0
                if fortes >= 3:
                    falando, frase, silencio = True, list(pre), 0
                continue
            frase.append(q)
            silencio = 0 if alto else silencio + 1
            longo = len(frase) >= self.quadros_trecho
            if (silencio >= self.quadros_silencio
                    or (longo and silencio >= self.quadros_pausa_curta)
                    or len(frase) >= max_quadros):
                fim_fala = time.time() - silencio * QUADRO_MS / 1000
                yield np.concatenate(frase).astype(np.float32) / 32768.0, fim_fala
                pre, frase, falando, silencio, fortes = [], [], False, 0, 0


class Piper:
    """Voz sintética local (genérica)."""

    def __init__(self, modelo_voz):
        from piper import PiperVoice
        self.voz = PiperVoice.load(str(modelo_voz))

    def sintetizar(self, texto):
        audio = b"".join(c.audio_int16_bytes for c in self.voz.synthesize(texto))
        return audio, self.voz.config.sample_rate


class VoiceStudio:
    """A sua voz clonada, gerada pela API local do VoiceStudio (precisa do app aberto)."""

    def __init__(self, voz, passos, url):
        self.voz, self.passos, self.url = voz, passos, url.rstrip("/")
        try:
            lista = json.load(urllib.request.urlopen(f"{self.url}/v1/audio/voices", timeout=5))["voices"]
        except OSError:
            raise SystemExit("Não consegui falar com o VoiceStudio. Abra o app e espere ele "
                             "terminar de carregar antes de rodar o tradutor.")
        perfis = {v["id"]: v["name"] for v in lista if v.get("type") == "profile"}
        if voz not in perfis:
            achados = [i for i, n in perfis.items() if n.lower() == voz.lower()]
            if not achados:
                raise SystemExit(f"Voz '{voz}' não encontrada no VoiceStudio. Disponíveis: "
                                 + ", ".join(perfis.values()))
            self.voz = achados[0]

    def aquecer(self):
        """Carrega o modelo na GPU agora (a primeira geração pode levar 1-2 min) e o mantém carregado."""
        inicio = time.time()
        self.sintetizar("Hi.")
        threading.Thread(target=self._manter_carregado, daemon=True).start()
        return time.time() - inicio

    def _manter_carregado(self):
        # o VoiceStudio descarrega o modelo após 15 min parado; uma frase curta a cada 10 min evita isso
        while True:
            time.sleep(600)
            try:
                self.sintetizar("Hi.")
            except Exception:
                pass

    def sintetizar(self, texto):
        corpo = json.dumps({
            "model": "tts-1", "input": texto, "voice": self.voz, "language": "en",
            "response_format": "wav", "num_step": self.passos,
        }).encode()
        req = urllib.request.Request(f"{self.url}/v1/audio/speech", data=corpo,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r, wave.open(io.BytesIO(r.read())) as w:
            return w.readframes(w.getnframes()), w.getframerate()


def dividir(texto, curto=25, longo=60, maximo=90):
    """Quebra a tradução em pedaços, para a voz do primeiro começar a tocar mais cedo."""
    partes = []
    for frase in re.split(r"(?<=[.!?;])\s+", texto):
        for p in re.split(r"(?<=,)\s+", frase) if len(frase) > longo else [frase]:
            # pedaço longo sem vírgula: quebra antes de uma conjunção que abre nova oração
            # ("and" só antes de pronome, para não separar "Vue and Angular")
            partes += (re.split(r"\s+(?=(?:but|so|because|which|where|when)\b"
                                r"|and\s+(?:I|we|it|they|he|she|you|then)\b)", p)
                       if len(p) > maximo else [p])
    juntas = []
    for p in partes:
        # pedaços muito curtos soam picados; junta com o anterior
        if juntas and len(juntas[-1]) < curto:
            juntas[-1] += " " + p
        else:
            juntas.append(p)
    return [j for j in juntas if j.strip()]


class Voz:
    """Gera a voz numa thread e toca em outra: o próximo pedaço é gerado enquanto o atual toca."""

    def __init__(self, motor, dispositivo, silenciar):
        self.motor = motor
        self.dispositivo = dispositivo
        self.silenciar = silenciar  # gravadores que ouviriam a voz traduzida enquanto ela toca
        self.textos = queue.Queue()
        self.audios = queue.Queue()
        threading.Thread(target=self._gerar, daemon=True).start()
        threading.Thread(target=self._tocar, daemon=True).start()

    def falar(self, texto, fim_fala):
        self.textos.put((texto, fim_fala))

    def _gerar(self):
        while True:
            texto, fim_fala = self.textos.get()
            for i, pedaco in enumerate(dividir(texto)):
                inicio = time.time()
                try:
                    audio, taxa = self.motor.sintetizar(pedaco)
                except Exception as e:
                    print(f"  (erro ao gerar a voz: {e})")
                    continue
                # só o primeiro pedaço mede o atraso desde que você parou de falar
                self.audios.put((audio, taxa, fim_fala if i == 0 else None, time.time() - inicio))

    def _tocar(self):
        while True:
            audio, taxa, fim_fala, t_voz = self.audios.get()
            if fim_fala is not None:
                print(f"  > tocando: voz {t_voz:.1f}s, atraso total {time.time() - fim_fala:.1f}s")
            cmd = ["pacat", "--playback", "--format=s16le", f"--rate={taxa}", "--channels=1"]
            if self.dispositivo:
                cmd.append(f"--device={self.dispositivo}")
            for g in self.silenciar:
                g.mudo.set()
            try:
                subprocess.run(cmd, input=audio, check=False)
            finally:
                if self.silenciar and self.audios.empty():
                    time.sleep(0.3)  # deixa o eco da sala morrer
                    for g in self.silenciar:
                        g.mudo.clear()


class InglesPortugues:
    """Traduz texto de inglês para português do Brasil, offline e no processador (OPUS-MT via CTranslate2)."""

    def __init__(self, pasta=MODELO_EN_PT):
        import ctranslate2
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(pasta / "sentencepiece.model"))
        self.modelo = ctranslate2.Translator(str(pasta / "model"), device="cpu", compute_type="int8",
                                             intra_threads=4)

    def traduzir(self, texto):
        # o modelo traduz uma frase por vez; frases juntas fazem ele pular partes
        frases = [f for f in re.split(r"(?<=[.!?])\s+", texto.strip()) if f]
        res = self.modelo.translate_batch([self.sp.encode(f, out_type=str) for f in frases], beam_size=2)
        return " ".join("".join(r.hypotheses[0]).replace("▁", " ").strip() for r in res)


def carregar_whisper(nome, dispositivo):
    from faster_whisper import WhisperModel
    if dispositivo in ("auto", "cuda"):
        try:
            # int8 ocupa metade da memória da placa, que é dividida com o VoiceStudio e o OBS
            return WhisperModel(nome, device="cuda", compute_type="int8_float16"), "GPU"
        except Exception as e:
            if dispositivo == "cuda":
                raise
            print(f"  (GPU indisponível: {e}; usando CPU)")
    return WhisperModel(nome, device="cpu", compute_type="int8"), "CPU"


def ouvir(whisper, audio, beam, idioma, tarefa):
    """tarefa 'translate': fala em `idioma` vira texto em inglês; 'transcribe': texto no próprio idioma."""
    segs, _ = whisper.transcribe(audio, language=idioma, task=tarefa,
                                 beam_size=beam, vad_filter=True,
                                 condition_on_previous_text=False)
    segs = [s for s in segs if s.no_speech_prob < 0.6]
    texto = " ".join(s.text.strip() for s in segs).strip()
    return "" if texto.lower().strip(" .!?,") in ALUCINACOES else texto


def escutar(fonte, gravador, pedidos):
    """Thread: manda cada frase de uma entrada para a fila única que o Whisper atende."""
    try:
        for audio, fim_fala in gravador.frases():
            if len(audio) >= TAXA_MIC * 0.4:
                pedidos.put((fonte, audio, fim_fala))
    except EOFError as e:
        pedidos.put(("erro", e, None))


def main():
    p = argparse.ArgumentParser(description="Fala em português, sai em inglês.")
    p.add_argument("--modelo", default="medium",
                   help="modelo do Whisper: small, medium, large-v3 (padrão: medium)")
    p.add_argument("--microfone-virtual", action="store_true",
                   help="manda a voz traduzida para um microfone virtual (Meet, Discord, Zoom)")
    p.add_argument("--ouvir-tambem", action="store_true",
                   help="com --microfone-virtual, também toca a tradução no seu fone")
    p.add_argument("--entrada", default=None, help="microfone (nome do pactl); padrão: o do sistema")
    p.add_argument("--saida", default=None, help="alto-falante (nome do pactl); padrão: o do sistema")
    p.add_argument("--voz", default=str(PASTA / "vozes" / "en_US-lessac-medium.onnx"),
                   help="voz do Piper, usada quando não há --voz-clonada")
    p.add_argument("--voz-clonada", default=None,
                   help="nome ou id de uma voz do VoiceStudio (ex.: felipe); o app precisa estar aberto")
    p.add_argument("--passos", type=int, default=16,
                   help="qualidade da voz clonada: mais passos = melhor e mais lento (padrão: 16)")
    p.add_argument("--voicestudio-url", default="http://127.0.0.1:3900")
    p.add_argument("--pausa", type=int, default=700,
                   help="ms de silêncio que encerram uma frase (padrão: 700)")
    p.add_argument("--trecho", type=int, default=6000,
                   help="ms de fala contínua a partir dos quais uma pausa curta já fecha o trecho (padrão: 6000)")
    p.add_argument("--beam", type=int, default=1,
                   help="busca do Whisper: 1 = mais rápido, 5 = um pouco mais preciso (padrão: 1)")
    p.add_argument("--sensibilidade", type=float, default=3.0,
                   help="quanto mais alto, menos sensível a ruído (padrão: 3.0)")
    p.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--arquivo", default=str(PASTA / "dublagem.txt"),
                   help="onde salvar o que foi dito (padrão: dublagem.txt)")
    p.add_argument("--meet", action="store_true",
                   help="também escuta o áudio do computador (a chamada) e mostra a tradução em português")
    p.add_argument("--meet-entrada", default="@DEFAULT_MONITOR@",
                   help="de onde vem o áudio da chamada (padrão: o que toca na saída padrão)")
    p.add_argument("--meet-idioma", default="en", help="idioma da outra pessoa (padrão: en)")
    p.add_argument("--meet-arquivo", default=str(PASTA / "meet.txt"),
                   help="onde salvar o que a outra pessoa disse (padrão: meet.txt)")
    a = p.parse_args()

    if a.voz_clonada:
        motor = VoiceStudio(a.voz_clonada, a.passos, a.voicestudio_url)
        print(f"Carregando a voz clonada '{a.voz_clonada}' no VoiceStudio (a primeira vez pode levar 1-2 min)...")
        print(f"  pronto ({motor.aquecer():.0f}s)")
    else:
        motor = Piper(a.voz)

    modulos = []
    try:
        print(f"Carregando Whisper '{a.modelo}'...")
        whisper, onde = carregar_whisper(a.modelo, a.dispositivo)
        print(f"  pronto ({onde})")

        saida = a.saida
        if a.microfone_virtual:
            modulos = criar_microfone_virtual()
            saida = SINK_VIRTUAL
            print("Microfone virtual criado: escolha 'Tradutor (microfone virtual)' no app da chamada.")
            if a.ouvir_tambem:
                modulos.insert(0, subprocess.check_output([
                    "pactl", "load-module", "module-loopback",
                    f"source={SINK_VIRTUAL}.monitor", "latency_msec=50",
                ], text=True).strip())

        mic = Gravador(a.entrada, a.pausa, a.sensibilidade, a.trecho)
        # com saída no alto-falante, o microfone escutaria a própria tradução
        silenciar = [] if a.microfone_virtual else [mic]

        meet = ingles_pt = None
        if a.meet:
            print("Carregando o tradutor inglês -> português...")
            ingles_pt = InglesPortugues()
            # o áudio do computador é digital e limpo: um limiar fixo baixo basta, sem calibrar
            meet = Gravador(a.meet_entrada, a.pausa, a.sensibilidade, a.trecho, limiar=300.0)
            # se a voz traduzida também toca na saída padrão, o ouvido do Meet a escutaria
            if not a.microfone_virtual or a.ouvir_tambem:
                silenciar.append(meet)

        voz = Voz(motor, saida, silenciar)

        print("Calibrando o ruído do ambiente, fique em silêncio 1 segundo...")
        ruido = mic.calibrar()
        print(f"  ruído {ruido:.0f}, limiar {mic.limiar:.0f}")

        pedidos = queue.Queue()
        threading.Thread(target=escutar, args=("eu", mic, pedidos), daemon=True).start()
        if meet:
            threading.Thread(target=escutar, args=("meet", meet, pedidos), daemon=True).start()
            print("Escutando também o áudio da chamada.")
        print("\nPode falar em português. Ctrl+C para sair.\n")

        with open(a.arquivo, "a", encoding="utf-8") as log, \
                open(a.meet_arquivo, "a", encoding="utf-8") as log_meet:
            while True:
                fonte, audio, fim_fala = pedidos.get()
                if fonte == "erro":
                    raise audio
                idioma, tarefa = ("pt", "translate") if fonte == "eu" else (a.meet_idioma, "transcribe")
                inicio = time.time()
                try:
                    texto = ouvir(whisper, audio, a.beam, idioma, tarefa)
                except RuntimeError as e:
                    if "out of memory" not in str(e) or onde == "CPU":
                        raise
                    print("  (a placa de vídeo ficou sem memória; passando o Whisper para o processador, "
                          "que é mais lento. Fechar o OBS libera memória.)")
                    whisper, onde = carregar_whisper(a.modelo, "cpu")
                    texto = ouvir(whisper, audio, a.beam, idioma, tarefa)
                if not texto:
                    continue

                if fonte == "meet":
                    pt = ingles_pt.traduzir(texto) if a.meet_idioma == "en" else texto
                    print(f"\n  MEET: {texto}\n     -> {pt}   ({time.time() - inicio:.1f}s)\n")
                    log_meet.write(f"{texto}\n-> {pt}\n\n")
                    log_meet.flush()
                    continue

                print(f"  EU: {texto}   (traduzir {time.time() - inicio:.1f}s)")
                log.write(texto + "\n")
                log.flush()
                voz.falar(texto, fim_fala)
    except KeyboardInterrupt:
        print("\nEncerrando.")
    except EOFError as e:
        print(f"\nErro: {e}", file=sys.stderr)
    finally:
        remover_modulos(modulos)


if __name__ == "__main__":
    main()
