# Dublê

Assistente local para conversas em inglês por vídeo, para quem fala português.

- **Entende a chamada:** transcreve o que a outra pessoa diz e mostra em inglês e português.
- **Sugere respostas:** gera 3 opções em inglês com o Gemini, baseadas no seu `perfil.txt`.
- **Fala por você:** traduz o que você diz e sai em inglês com a sua voz clonada, por um microfone virtual que você escolhe no Meet, Zoom ou Discord. Clicar numa sugestão também faz ela ser falada.
- **Volta para a sua voz:** com a tradução desligada, o microfone virtual passa a sua voz normal, sem precisar trocar nada na chamada.

Transcrição, tradução e voz rodam no seu computador. Só as sugestões usam a internet: a conversa recente e o seu perfil são enviados ao Gemini quando você pede uma sugestão.

## Requisitos

- Linux com PulseAudio (`parec`, `pacat` e `pactl`). Testado no Ubuntu 22.04 com Python 3.10.
- GPU NVIDIA é recomendada. Testado com uma RTX 3050 de 8 GB; sem GPU, roda no processador, bem mais devagar.
- Para usar a sua voz: [VoiceStudio](https://github.com/debpalash/VoiceStudio) aberto, com a voz criada nele. Sem ele, a voz é a genérica do Piper.

## Instalação

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Voz genérica
mkdir -p vozes
.venv/bin/python -m piper.download_voices --download-dir vozes en_US-lessac-medium

# Tradutor inglês -> português
mkdir -p modelos
curl -L -o modelos/en_pb.zip https://argos-net.com/v1/translate-en_pb-1_9.argosmodel
unzip -q modelos/en_pb.zip -d modelos && rm modelos/en_pb.zip

# Configuração
cp .env.example .env              # coloque a GEMINI_API_KEY
cp perfil.exemplo.txt perfil.txt  # preencha com os seus dados
```

## Como usar

```bash
./painel.sh
```

Abra http://127.0.0.1:8765 e, na chamada, escolha o microfone **Tradutor (microfone virtual)**. A saída de som da chamada precisa ser a saída padrão do sistema. Use fone de ouvido: pela caixa de som, o microfone capta a chamada, e o filtro de eco descarta o que você falar enquanto a outra pessoa estiver falando.

Opções: `./painel.sh --help`. Para só traduzir a voz, sem painel: `./tradutor.sh --voz-clonada <nome> --microfone-virtual`.

## Uso responsável

Clone apenas a sua própria voz. A conversa fica salva em `conversa.txt` no seu computador, então avise as outras pessoas da chamada quando a transcrição ou a voz sintética exigirem isso.

## Créditos

[faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Piper](https://github.com/OHF-Voice/piper1-gpl), [VoiceStudio](https://github.com/debpalash/VoiceStudio) e [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) (Tiedemann e Thottingal, CC-BY 4.0, empacotado pelo Argos Translate).
