"""
Módulo Core do Bot Telegram Multimodal com IA.
Implementa a arquitetura orientada a objetos (POO) com herança,
encapsulamento e polimorfismo.
"""

from .bot_base import BotTelegram
from .bot_imagem import BotImagem
from .bot_audio import BotAudio
from .bot_completo import BotMultimodal

__all__ = ["BotTelegram", "BotImagem", "BotAudio", "BotMultimodal"]
