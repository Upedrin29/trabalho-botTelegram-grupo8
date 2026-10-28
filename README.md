# 🤖 Bot do Telegram Multimodal com Inteligência Artificial (POO)

Um bot completo para o Telegram desenvolvido em **Python** seguindo estritamente o paradigma da **Programação Orientada a Objetos (POO)**. O bot é capaz de receber e processar dinamicamente **imagens**, **áudios** e **textos**, aplicando modelos de Inteligência Artificial:

1. **Visão Computacional:** Classificação e detecção de objetos em tempo real utilizando **YOLOv8** (`ultralytics`) e renderização visual com caixas delimitadoras personalizadas e sobreposição gráfica em **OpenCV**.
2. **Processamento de Áudio e Fala (Speech-to-Text & NLP):** Conversão de áudios (formato OGG/Opus do Telegram para PCM) com `pydub`/`ffmpeg`, transcrição de voz com `SpeechRecognition` e classificação do conteúdo falado com análise de sentimento (Positivo, Neutro ou Negativo) e intenção da fala.

---

## 🏛️ Arquitetura Orientada a Objetos (POO)

O projeto foi concebido para evidenciar os pilares fundamentais da orientação a objetos:

```
                  ┌────────────────────────────────────────┐
                  │              BotTelegram               │
                  │  (Gerencia API, Dispatcher, __token)   │
                  └───────────────────┬────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     ┌───────────────────────────┐         ┌───────────────────────────┐
     │         BotImagem         │         │         BotAudio          │
     │  (YOLOv8 + OpenCV Vision) │         │    (STT + NLP Classifier) │
     └─────────────┬─────────────┘         └─────────────┬─────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │       BotMultimodal       │
                        │    (Herança Múltipla)     │
                        └───────────────────────────┘
```

### 1. Encapsulamento
- Os atributos sensíveis, especialmente o token da API do Telegram (`__token`), são declarados como estritamente privados com *name mangling*.
- O acesso externo é restrito e auditado por propriedades seguras (como `token_mascarado`, que exibe apenas prefixo e sufixo como `123456...wxyz`).
- O token nunca é exposto em mensagens de erro ou logs públicos.

### 2. Herança
- `BotTelegram`: Classe base abstrata/gerenciadora que encapsula a infraestrutura de rede, o ciclo de vida do polling do `python-telegram-bot`, comandos universais (`/start`, `/help`, `/stats`) e a identificação de tipos de mensagens.
- `BotImagem(BotTelegram)`: Especialista em visão computacional; carrega o modelo YOLOv8, decodifica matrizes BGR no OpenCV e constrói imagens anotadas com caixas e cabeçalhos visuais.
- `BotAudio(BotTelegram)`: Especialista em áudio e fala; decodifica containers OGG/Opus do Telegram, extrai streams PCM 16kHz mono, aciona o Speech-to-Text e classifica o texto falado.
- `BotMultimodal(BotImagem, BotAudio)`: Classe integrada que herda de ambas as especialidades (herança múltipla cooperativa via `super().__init__`), oferecendo ao usuário um assistente unificado.

### 3. Polimorfismo
- O método central `despachar_mensagem(update, context)` da classe base inspeciona a mensagem recebida e aciona dinamicamente os métodos virtuais:
  - `processar_imagem()`: sobrescrito em `BotImagem`.
  - `processar_audio()`: sobrescrito em `BotAudio`.
  - `processar_texto()`: implementado com cálculos estatísticos e orientações inteligentes.
- Cada classe especialista responde ao mesmo evento de forma distinta e especializada.

---

## 📁 Estrutura do Projeto

```
trabalho ENE/
├── .env.example              # Modelo para configuração das variáveis de ambiente
├── requirements.txt          # Dependências do projeto (YOLO, OpenCV, Telegram, etc.)
├── bot.py                    # Script principal de inicialização e interface CLI
├── README.md                 # Documentação completa
├── core/
│   ├── __init__.py           # Exportação dos componentes da arquitetura
│   ├── bot_base.py           # Classe base BotTelegram (encapsulamento e polimorfismo)
│   ├── bot_imagem.py         # Subclasse BotImagem (YOLOv8 + OpenCV)
│   ├── bot_audio.py          # Subclasse BotAudio (Speech-to-Text + Classificador NLP)
│   └── bot_completo.py       # Subclasse BotMultimodal (unificação e comandos extras)
└── tests/
    └── test_local.py         # Suíte de testes offline e validação de POO
```

---

## 🚀 Como Executar

### 1. Ativação do Ambiente Virtual
No terminal PowerShell ou CMD:
```powershell
.\.venv\Scripts\activate
```

### 2. Configuração do Token do Telegram
Obtenha o token do seu bot conversando com o [@BotFather](https://t.me/BotFather) no Telegram.

Crie um arquivo `.env` na raiz do projeto (ou copie o `.env.example`):
```bash
copy .env.example .env
```
Edite o arquivo `.env`:
```env
TELEGRAM_BOT_TOKEN=seu_token_aqui_sem_aspas
YOLO_MODEL=yolov8n.pt
AUDIO_LANGUAGE=pt-BR
```

### 3. Execução dos Testes Locais (Sem precisar de conexão com Telegram)
Para validar o modelo YOLO, o desenho OpenCV, o classificador de áudio, a herança e o encapsulamento:
```bash
python bot.py --test-local
```
ou diretamente:
```bash
python tests/test_local.py
```

### 4. Inicialização do Bot
Inicie o bot no modo multimodal padrão:
```bash
python bot.py
```

#### Modos de Execução Alternativos:
- **Apenas Visão (YOLO):** `python bot.py --mode imagem`
- **Apenas Áudio (Fala):** `python bot.py --mode audio`
- **Passando token direto via linha de comando:** `python bot.py --token "SEU_TOKEN"`

---

## 💬 Interagindo com o Bot no Telegram

| Entrada | Ação do Bot | Resultado |
|---|---|---|
| **Foto / Imagem** | Processada com **YOLOv8** e **OpenCV** | Retorna a foto com caixas coloridas demarcando cada objeto e um relatório detalhado com contagem por classe e confiança. |
| **Áudio / Mensagem de Voz** | Convertido para WAV, transcrito via **Speech-to-Text** e classificado | Retorna o texto transcrito, classificação de sentimento (Positivo/Negativo/Neutro), intenção (Dúvida, Suporte, Elogio, etc.) e velocidade de fala (WPM). |
| **Texto Comum** | Analisado linguisticamente | Métricas de palavras, caracteres e orientações para testar as IAs. |
| `/start` | Boas-vindas | Apresentação e menu inicial. |
| `/help` | Guia completo | Explicação detalhada de todos os recursos. |
| `/vision` | Informações de Visão | Lista de categorias do dataset COCO detectáveis pelo YOLO. |
| `/voice` | Informações de Fala | Instruções de pronúncia e exemplos de frases para testar. |
| `/stats` | Estatísticas | Total de imagens, áudios e textos processados na sessão. |
| `/about` | Arquitetura POO | Mostra o grafo de herança, MRO e encapsulamento do sistema. |
