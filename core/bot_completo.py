"""
Módulo integrador do Bot Telegram Multimodal.
Implementa a classe BotMultimodal através de herança múltipla de BotImagem e BotAudio,
unificando a Visão Computacional (YOLOv8 + OpenCV) e o Processamento de Fala/NLP.
"""

from __future__ import annotations
import logging
from typing import Optional, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from .bot_imagem import BotImagem
from .bot_audio import BotAudio

logger = logging.getLogger("BotMultimodal")


class BotMultimodal(BotImagem, BotAudio):
    """
    Classe unificada que herda de BotImagem e BotAudio.
    
    Demonstra com excelência os princípios da Programação Orientada a Objetos:
    1. Herança: Herda capacidades de visão computacional de BotImagem e
       capacidades de processamento de fala/NLP de BotAudio, ambas filhas de BotTelegram.
    2. Polimorfismo: O despacho polimórfico na classe base executa os métodos
       sobrescritos em cada classe especialista dinamicamente de acordo com o tipo
       de mensagem (foto, voz/áudio, texto).
    3. Encapsulamento: O token da API e as chaves de configuração permanecem
       estritamente encapsulados na raiz da hierarquia em BotTelegram.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        modelo_yolo: str = "yolov8n.pt",
        confianca_minima: float = 0.25,
        idioma_reconhecimento: str = "pt-BR",
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o Bot Multimodal combinando Visão Computacional e Reconhecimento de Fala.
        
        :param token: Token de autenticação da API do Telegram.
        :param modelo_yolo: Nome ou caminho do modelo YOLO (ex: 'yolov8n.pt').
        :param confianca_minima: Limiar de confiança do detector YOLO.
        :param idioma_reconhecimento: Idioma para transcrição de áudio (padrão: 'pt-BR').
        """
        logger.info("Inicializando BotMultimodal com herança múltipla cooperativa...")
        super().__init__(
            token=token,
            modelo_yolo=modelo_yolo,
            confianca_minima=confianca_minima,
            idioma_reconhecimento=idioma_reconhecimento,
            **kwargs,
        )
        logger.info("BotMultimodal pronto para operar!")

    # -------------------------------------------------------------------------
    # Extensões e Novos Recursos Criativos
    # -------------------------------------------------------------------------

    async def comando_vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /vision: Detalha o modelo de Visão Computacional e classes detectáveis."""
        mensagem = (
            "👁️ *Central de Visão Computacional (YOLOv8 & OpenCV)*\n\n"
            f"• **Modelo Ativo:** `{self._modelo_yolo_nome}`\n"
            f"• **Limiar de Confiança:** `{self._confianca_minima * 100:.0f}%`\n"
            f"• **Pipeline:** Decodificação OpenCV -> Inferência YOLOv8 -> "
            f"Desenho de Caixas & Labels com OpenCV -> Codificação JPEG\n\n"
            "📦 *Exemplos de Classes Detectáveis (Dataset COCO - 80 classes):*\n"
            "• Pessoas, bicicletas, carros, motos, aviões, ônibus, trens, barcos;\n"
            "• Semáforos, placas de trânsito, hidrantes, bancos de parque;\n"
            "• Animais: pássaros, gatos, cachorros, cavalos, ovelhas, vacas, elefantes, ursos;\n"
            "• Objetos: mochilas, guarda-chuvas, garrafas, copos, garfos, facas, tigelas;\n"
            "• Alimentos: maçãs, bananas, sanduíches, laranjas, pizzas, bolos;\n"
            "• Móveis e Eletrônicos: cadeiras, sofás, camas, TVs, notebooks, mouses, celulares, micro-ondas, livros.\n\n"
            "📸 *Basta enviar qualquer foto para ver o modelo em ação!*"
        )
        await update.message.reply_text(mensagem, parse_mode="Markdown")

    async def comando_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /voice: Detalha o motor de Áudio, Speech-to-Text e Classificador NLP."""
        mensagem = (
            "🎙️ *Central de Reconhecimento de Fala & NLP*\n\n"
            f"• **Idioma Ativo:** `{self._idioma}`\n"
            f"• **Conversor de Áudio:** PyDub com motor FFmpeg integrado\n"
            f"• **Engine Speech-to-Text:** SpeechRecognition (Google API / PCM 16kHz Mono)\n"
            f"• **Classificador Semântico:** Análise de Sentimento (Léxico pt-BR) e Intenção via RegEx/NLP\n\n"
            "💡 *Como testar:*\n"
            "• Grave uma mensagem de voz no Telegram dizendo algo como:\n"
            "  - *\"Adorei esse bot, o atendimento foi excelente e muito rápido!\"* (Sentimento Positivo)\n"
            "  - *\"Estou com um erro grave no meu sistema, preciso de ajuda urgente!\"* (Suporte / Negativo)\n"
            "  - *\"Como faço para calcular a média ponderada?\"* (Pergunta / Dúvida)\n\n"
            "O bot transcreverá a fala e classificará o sentimento e a intenção em segundos!"
        )
        await update.message.reply_text(mensagem, parse_mode="Markdown")

    async def comando_about(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comando /about: Detalhes da arquitetura de Programação Orientada a Objetos."""
        mro_classes = [c.__name__ for c in self.__class__.__mro__ if c.__name__ not in ("object", "ABC")]
        mensagem = (
            "🏛️ *Arquitetura Orientada a Objetos (POO):*\n\n"
            "1. **Herança e Hierarquia de Classes:**\n"
            "   `BotTelegram` (Base)\n"
            "      ├── `BotImagem` (Visão Computacional: YOLOv8 + OpenCV)\n"
            "      └── `BotAudio` (Áudio: Conversão + STT + Classificador NLP)\n"
            "            └── `BotMultimodal` (Unificação de ambas as especialidades)\n\n"
            f"2. **Resolução de Métodos (MRO):**\n"
            f"   `{' -> '.join(mro_classes)}`\n\n"
            "3. **Encapsulamento:**\n"
            f"   O atributo de credencial sensível (`__token`) é estritamente privado, "
            f"com acesso seguro mascarado (`{self.token_mascarado}`).\n\n"
            "4. **Polimorfismo:**\n"
            "   O método `despachar_mensagem` inspeciona polimorficamente o tipo da mensagem recebida "
            "e direciona para as implementações especializadas `processar_imagem`, `processar_audio` "
            "e `processar_texto`."
        )
        await update.message.reply_text(mensagem, parse_mode="Markdown")

    def registrar_handlers(self, app: Application) -> None:
        """
        Sobrescreve o registro de handlers para incluir os comandos criativos adicionais.
        """
        super().registrar_handlers(app)
        app.add_handler(CommandHandler("vision", self.comando_vision))
        app.add_handler(CommandHandler("voice", self.comando_voice))
        app.add_handler(CommandHandler("about", self.comando_about))
