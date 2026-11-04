"""Painel da chamada: mostra o que a outra pessoa disse (inglês + português), sugere respostas
com o Gemini e, quando ligado, traduz o que você fala para inglês.

Abre em http://127.0.0.1:8765. A voz em inglês (a sua fala traduzida ou uma sugestão escolhida)
sai pelo microfone virtual "Tradutor (microfone virtual)", que você escolhe no Meet.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tradutor import (PASTA, SINK_VIRTUAL, TAXA_MIC, Gravador, InglesPortugues, Piper, VoiceStudio, Voz,
                      carregar_whisper, criar_microfone_virtual, escutar, ouvir, remover_modulos)

PERFIL = PASTA / "perfil.txt"
PAGINA = PASTA / "painel.html"


def carregar_env(caminho=PASTA / ".env"):
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#") and "=" in linha:
            chave, valor = linha.split("=", 1)
            os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


def nome_do_perfil():
    """Primeiro nome da linha 'Nome:' do perfil.txt, para a IA saber com quem está falando."""
    if PERFIL.exists():
        achado = re.search(r"^Nome:[ \t]*(\S+)", PERFIL.read_text(encoding="utf-8"), re.MULTILINE)
        if achado:
            return achado.group(1)
    return "the user"


class Conversa:
    """Histórico da chamada e os navegadores conectados que recebem cada novidade."""

    def __init__(self, arquivo):
        self.itens = []
        self.clientes = []
        self.trava = threading.Lock()
        self.microfone = False
        self.arquivo = arquivo
        self.ingles_pt = None  # carregado depois do servidor subir, para o painel abrir logo

    def traduzir(self, en):
        return self.ingles_pt.traduzir(en) if self.ingles_pt else ""

    def adicionar(self, tipo, en, pt=""):
        with self.trava:
            item = {"id": len(self.itens), "tipo": tipo, "en": en, "pt": pt,
                    "hora": time.strftime("%H:%M:%S"), "sugestoes": None}
            self.itens.append(item)
        with open(self.arquivo, "a", encoding="utf-8") as f:
            f.write(f"[{item['hora']}] {'MEET' if tipo == 'meet' else 'EU'}: {en}\n"
                    + (f"    -> {pt}\n" if pt else ""))
        self.enviar("item", item)
        return item

    def enviar(self, evento, dados):
        msg = f"event: {evento}\ndata: {json.dumps(dados, ensure_ascii=False)}\n\n".encode()
        with self.trava:
            for c in self.clientes:
                c.put(msg)

    def contexto(self, ate_id, n=12):
        """As últimas falas até a pergunta escolhida, para a IA entender a conversa."""
        linhas = []
        for it in self.itens[max(0, ate_id - n + 1):ate_id + 1]:
            quem = "Interviewer/other person" if it["tipo"] == "meet" else nome_do_perfil()
            linhas.append(f"{quem}: {it['en']}")
        return "\n".join(linhas)


class PassagemMic:
    """Com a tradução desligada, manda o seu microfone direto para o microfone virtual,
    para o Meet não precisar trocar de microfone."""

    def __init__(self, microfone):
        self.microfone = microfone or subprocess.check_output(
            ["pactl", "get-default-source"], text=True).strip()
        self.modulo = None
        self.trava = threading.Lock()

    def ligar(self, ligado):
        with self.trava:
            if ligado and self.modulo is None:
                self.modulo = subprocess.check_output([
                    "pactl", "load-module", "module-loopback", f"source={self.microfone}",
                    f"sink={SINK_VIRTUAL}", "latency_msec=30", "source_dont_move=true",
                ], text=True).strip()
            elif not ligado and self.modulo is not None:
                remover_modulos([self.modulo])
                self.modulo = None


class Sugestor:
    """Pede ao Gemini respostas curtas em inglês, baseadas no perfil.txt."""

    def __init__(self):
        from google import genai
        from google.genai import types
        chave = os.environ.get("GEMINI_API_KEY", "")
        if not chave:
            raise RuntimeError("GEMINI_API_KEY vazia: coloque a chave no arquivo .env")
        # 20 s por tentativa: em horário de pico a API pode prender a resposta por minutos
        self.cliente = genai.Client(api_key=chave, http_options=types.HttpOptions(timeout=20000))
        # vários modelos separados por vírgula: se um estiver sobrecarregado, tenta o próximo
        self.modelos = [m.strip() for m in os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").split(",")
                        if m.strip()]
        self.modelo = self.modelos[0]

    def sugerir(self, contexto, pergunta):
        from google.genai import types
        perfil = PERFIL.read_text(encoding="utf-8") if PERFIL.exists() else "(sem perfil)"
        nome = nome_do_perfil()
        prompt = f"""You help {nome}, a Brazilian professional, during a live English conversation
(often a job interview). They will READ your suggestions out loud, so write natural, simple,
spoken English: short sentences, no jargon he would stumble on, no bullet points.

Use ONLY facts from his profile below. Never invent companies, numbers, projects or skills.
If the profile does not cover what was asked, suggest an honest answer that does not
claim specifics (or briefly asks for clarification).

Give exactly 3 different options:
1. short and direct (1-2 sentences)
2. more complete (3-4 sentences, with one concrete example from the profile if possible)
3. a different angle (e.g. honest/humble, or a clarifying question back)

For each option also give a Brazilian Portuguese translation so he understands what he is saying.

=== Profile of {nome} (written in Portuguese) ===
{perfil}

=== Conversation so far ===
{contexto}

=== Answer this last message from the other person ===
{pergunta}

Return JSON only: {{"respostas": [{{"en": "...", "pt": "..."}}, ...]}}"""
        erros = []
        for modelo in self.modelos:
            config = {"response_mime_type": "application/json", "temperature": 0.7}
            # pensar pouco antes de responder: bem mais rápido, e respostas curtas não precisam disso
            if "2.5" in modelo:
                config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            elif "gemini-3" in modelo:
                config["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
            try:
                r = self.cliente.models.generate_content(
                    model=modelo, contents=prompt, config=types.GenerateContentConfig(**config))
                self.modelo = modelo
                return json.loads(r.text)["respostas"]
            except Exception as e:
                # sobrecarga, limite ou demora: tenta o próximo modelo da lista
                if not any(c in str(e) for c in ("503", "504", "429", "timed out", "Timeout")):
                    raise
                erros.append(f"{modelo}: {str(e)[:80]}")
        raise RuntimeError("todos os modelos estão ocupados agora, tente de novo: " + " | ".join(erros))


def criar_servidor(conversa, sugestor, voz, passagem, porta):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, codigo, dados):
            corpo = json.dumps(dados, ensure_ascii=False).encode()
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def _corpo(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/":
                corpo = PAGINA.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)
            elif self.path == "/eventos":
                self._eventos()
            else:
                self.send_error(404)

        def _eventos(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            fila = queue.Queue()
            with conversa.trava:
                inicial = {"itens": conversa.itens, "microfone": conversa.microfone,
                           "sugestor": sugestor is not None, "voz": voz is not None}
                conversa.clientes.append(fila)
            try:
                self.wfile.write(f"event: inicio\ndata: {json.dumps(inicial, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = fila.get(timeout=15)
                    except queue.Empty:
                        msg = b": ping\n\n"  # mantém a conexão viva
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with conversa.trava:
                    conversa.clientes.remove(fila)

        def do_POST(self):
            dados = self._corpo()
            if self.path == "/microfone":
                conversa.microfone = bool(dados.get("ligado"))
                if passagem:
                    # tradução ligada: o Meet ouve só a voz traduzida; desligada: ouve você direto
                    passagem.ligar(not conversa.microfone)
                conversa.enviar("microfone", {"ligado": conversa.microfone})
                self._json(200, {"ligado": conversa.microfone})
            elif self.path == "/pergunta":
                # pergunta digitada à mão (quando o áudio falhar)
                en = dados.get("texto", "").strip()
                if en:
                    conversa.adicionar("meet", en, conversa.traduzir(en))
                self._json(200, {})
            elif self.path == "/sugerir":
                self._sugerir(dados.get("id"))
            elif self.path == "/falar":
                # fala uma sugestão escolhida no Meet, com a voz traduzida
                texto = dados.get("texto", "").strip()
                if voz is None:
                    return self._json(400, {"erro": "voz desligada (painel iniciado com --sem-voz)"})
                if texto:
                    voz.falar(texto, None)
                self._json(200, {})
            else:
                self.send_error(404)

        def _sugerir(self, item_id):
            if sugestor is None:
                return self._json(400, {"erro": "Coloque a GEMINI_API_KEY no arquivo .env e reinicie o painel."})
            try:
                item = conversa.itens[int(item_id)]
            except (TypeError, ValueError, IndexError):
                return self._json(404, {"erro": "fala não encontrada"})
            inicio = time.time()
            try:
                respostas = sugestor.sugerir(conversa.contexto(item["id"]), item["en"])
            except Exception as e:
                return self._json(502, {"erro": f"Gemini: {e}"})
            item["sugestoes"] = respostas
            conversa.enviar("sugestoes", {"id": item["id"], "sugestoes": respostas,
                                          "tempo": round(time.time() - inicio, 1)})
            self._json(200, {"sugestoes": respostas})

    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    servidor.daemon_threads = True
    return servidor


def main():
    p = argparse.ArgumentParser(description="Painel da chamada com tradução e sugestões de resposta.")
    p.add_argument("--porta", type=int, default=8765)
    p.add_argument("--modelo", default="medium", help="modelo do Whisper (padrão: medium)")
    p.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--entrada", default=None, help="seu microfone (padrão: o do sistema)")
    p.add_argument("--meet-entrada", default="@DEFAULT_MONITOR@",
                   help="de onde vem o áudio da chamada (padrão: o que toca na saída padrão)")
    p.add_argument("--pausa", type=int, default=700, help="ms de silêncio que encerram uma frase")
    p.add_argument("--trecho", type=int, default=6000)
    p.add_argument("--sensibilidade", type=float, default=3.0)
    p.add_argument("--arquivo", default=str(PASTA / "conversa.txt"),
                   help="onde salvar a conversa (padrão: conversa.txt)")
    p.add_argument("--voz-clonada", default=None,
                   help="nome de uma voz do VoiceStudio (ex.: minha_voz); padrão: VOZ_CLONADA do .env, "
                        "ou a voz do Piper se não houver")
    p.add_argument("--passos", type=int, default=None,
                   help="qualidade da voz clonada: menos = mais rápido (padrão: VOZ_PASSOS do .env, ou 16)")
    p.add_argument("--voicestudio-url", default="http://127.0.0.1:3900")
    p.add_argument("--voz", default=str(PASTA / "vozes" / "en_US-lessac-medium.onnx"),
                   help="voz do Piper, usada quando não há --voz-clonada")
    p.add_argument("--sem-voz", action="store_true", help="só texto: não fala nada no Meet")
    a = p.parse_args()

    carregar_env()
    a.voz_clonada = a.voz_clonada or os.environ.get("VOZ_CLONADA") or None
    a.passos = a.passos or int(os.environ.get("VOZ_PASSOS") or 16)
    modulos, voz, passagem = [], None, None
    if not a.sem_voz:
        if a.voz_clonada:
            motor = VoiceStudio(a.voz_clonada, a.passos, a.voicestudio_url)
            print(f"Carregando a voz clonada '{a.voz_clonada}' no VoiceStudio (a primeira vez pode levar 1-2 min)...")
            print(f"  pronto ({motor.aquecer():.0f}s)")
        else:
            motor = Piper(a.voz)
            print("Usando a voz genérica do Piper (defina VOZ_CLONADA no .env para usar a sua).")
        modulos = criar_microfone_virtual()
        # a voz vai só para o microfone virtual: nem o seu microfone nem o áudio da chamada a escutam
        voz = Voz(motor, SINK_VIRTUAL, [])
        passagem = PassagemMic(a.entrada)
        passagem.ligar(True)  # começa com a tradução desligada: o Meet ouve você direto
        print("Voz ligada: no Meet, escolha o microfone 'Tradutor (microfone virtual)'.")
        print("  Com 'Traduzir minha fala' desligado, o Meet ouve a sua voz normal por ele.")
    try:
        sugestor = Sugestor()
        print(f"Sugestões: Gemini ({sugestor.modelo})")
    except Exception as e:
        sugestor = None
        print(f"Sugestões desligadas: {e}")

    conversa = Conversa(a.arquivo)
    servidor = criar_servidor(conversa, sugestor, voz, passagem, a.porta)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    print(f"\nPainel: http://127.0.0.1:{a.porta}\n")

    print(f"Carregando Whisper '{a.modelo}'...")
    whisper, onde = carregar_whisper(a.modelo, a.dispositivo)
    print(f"  pronto ({onde})")
    conversa.ingles_pt = InglesPortugues()

    mic = Gravador(a.entrada, a.pausa, a.sensibilidade, a.trecho)
    meet = Gravador(a.meet_entrada, a.pausa, a.sensibilidade, a.trecho, limiar=300.0)
    print("Calibrando o ruído do microfone, fique em silêncio 1 segundo...")
    mic.calibrar()

    pedidos = queue.Queue()
    threading.Thread(target=escutar, args=("eu", mic, pedidos), daemon=True).start()
    threading.Thread(target=escutar, args=("meet", meet, pedidos), daemon=True).start()
    print("Escutando a chamada. Ctrl+C para sair.\n")

    try:
        while True:
            fonte, audio, fim_fala = pedidos.get()
            if fonte == "erro":
                raise audio
            if fonte == "eu":
                if not conversa.microfone:
                    continue  # botão "Traduzir minha fala" desligado
                inicio_fala = fim_fala - len(audio) / TAXA_MIC
                if meet.teve_som(inicio_fala, fim_fala):
                    # sem fone, o microfone capta a chamada pela caixa de som: não traduzir isso
                    print("  (ignorei uma fala do microfone durante a fala da outra pessoa: possível eco)")
                    continue
            try:
                if fonte == "meet":
                    en = ouvir(whisper, audio, 1, "en", "transcribe")
                    if en:
                        conversa.adicionar("meet", en, conversa.traduzir(en))
                else:
                    en = ouvir(whisper, audio, 1, "pt", "translate")
                    if en:
                        conversa.adicionar("eu", en)
                        if voz:
                            voz.falar(en, fim_fala)
            except RuntimeError as e:
                if "out of memory" not in str(e) or onde == "CPU":
                    raise
                print("  (placa de vídeo sem memória; passando o Whisper para o processador)")
                whisper, onde = carregar_whisper(a.modelo, "cpu")
    except KeyboardInterrupt:
        print("\nEncerrando.")
    except EOFError as e:
        print(f"\nErro: {e}")
    finally:
        servidor.shutdown()
        if passagem:
            passagem.ligar(False)
        remover_modulos(modulos)


if __name__ == "__main__":
    main()
