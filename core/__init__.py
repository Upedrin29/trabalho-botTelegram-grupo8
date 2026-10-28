"""
Módulo Core do Bot Telegram Multimodal com IA.
Implementa a arquitetura orientada a objetos (POO) com herança,
encapsulamento e polimorfismo.
"""

# Sem esse __init__.py, o bot.py teria que usar o caminho completo de cada arquivo:
# Ou seja, o __init__.py funciona como uma fachada: esconde a organização interna dos arquivos e oferece um ponto de acesso único e limpo.
# Se você renomear bot_completo.py no futuro, só precisa ajustar esta linha, e o bot.py continua igual.


# a ordem dos importes importa para herança
from .bot_base import BotTelegram
from .bot_imagem import BotImagem
from .bot_audio import BotAudio
from .bot_completo import BotMultimodal


# Uma lista com os nomes que formam a interface pública do pacote
# Controla o que é importado em from core import *
# Documenta, para quem lê e para as ferramentas do editor, quais itens são “oficiais”.
__all__ = ["BotTelegram", "BotImagem", "BotAudio", "BotMultimodal"]
