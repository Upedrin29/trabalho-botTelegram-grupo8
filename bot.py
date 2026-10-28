#!/usr/bin/env python3
"""
Ponto de entrada principal do Bot Telegram com Inteligência Artificial.

Demonstração prática de Programação Orientada a Objetos (POO):
- Herança (BotTelegram -> BotImagem, BotAudio -> BotMultimodal)
- Encapsulamento (Token de API estritamente protegido)
- Polimorfismo (Tratamento dinâmico de Texto, Imagem e Áudio)
- Visão Computacional (YOLOv8 + OpenCV)
- Reconhecimento de Fala e Classificação Semântica (SpeechRecognition + NLP)
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env caso exista
load_dotenv()


# Configuração de logging estruturado
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",       # format: define o padrão de cada linha, %(asctime)s → data e hora, %(levelname)s → nível de severidade, %(name)s → nome do módulo, %(message)s → texto da mensagem
    level=logging.INFO,                                                  # mostra mensagens de INFO
)

logger = logging.getLogger("Main")                                       # Cria um logger chamado "Main", que aparece no %(name)s. Serve para identificar que a mensagem veio deste arquivo.



from core import BotTelegram, BotImagem, BotAudio, BotMultimodal



#------------- Define a função que lê o que o usuário digitou no terminal -----------------

def parse_argumentos() -> argparse.Namespace:  # O retorno da função é um objeto onde cada argumento vira um atributo


    """ cria e configura o leitor de argumentos do terminal"""
    parser = argparse.ArgumentParser(
        description="Bot Telegram com IA (YOLOv8 + OpenCV + Speech-to-Text) estruturado em POO."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token da API do Telegram (ou configure TELEGRAM_BOT_TOKEN no .env).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multimodal", "imagem", "audio"],
        default="multimodal",
        help="Modo de operação do bot: 'multimodal' (padrão), 'imagem' (apenas YOLO) ou 'audio' (apenas fala).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        help="Modelo YOLO a ser utilizado (padrão: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confiança mínima para detecções YOLO (padrão: 0.25).",
    )
    parser.add_argument(
        "--test-local",
        action="store_true",                                                # flag que será true caso rode o teste local e false caso contrário
        help="Executa a suíte de testes locais sem conectar ao Telegram.",
    )
    return parser.parse_args()



# ---------------- Define a função principal que inicia o bot ---------------

def main() -> None:
    """Função de inicialização do sistema."""
    args = parse_argumentos()

    # Caso a flag --test-local seja acionada, roda os testes e finaliza
    if args.test_local:
        logger.info("Executando suíte de testes locais...")
        import subprocess
        codigo = subprocess.call([sys.executable, "tests/test_local.py"])   # executa outro script como um processo separado.
        sys.exit(codigo)                                                    # finaliza o programa passando o código de saída do teste (0 = sucesso).

    # Validação prévia de token
    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not token.strip():
        print("\n" + "=" * 70)
        print(" ATENÇÃO: TOKEN DO TELEGRAM NÃO CONFIGURADO!")
        print("\n DICA: Você pode validar todo o funcionamento dos modelos de IA localmente")
        print("   sem precisar de token executando o comando:")
        print("   python bot.py --test-local")
        print("=" * 70 + "\n")
        sys.exit(1)                                                         # Encerra o programa com código de erro 1


    # Instanciação do bot (polimorfismo) baseada no modo escolhido
    logger.info("Instanciando bot no modo: %s", args.mode.upper())
    
    bot_instancia: BotTelegram                                              # anotação de tipo: declara que a variável guardará um BotTelegram (ou qualquer filha dele).
    if args.mode == "imagem":
        bot_instancia = BotImagem(                                          # cria instância de BotImagem
            token=token,
            modelo_yolo=args.model,
            confianca_minima=args.conf,
        )
    elif args.mode == "audio":                                              # cria instância de BotAudio
        bot_instancia = BotAudio(
            token=token,
            idioma_reconhecimento=os.getenv("AUDIO_LANGUAGE", "pt-BR"),
        )
    else:  # Modo Multimodal (Padrão)
        bot_instancia = BotMultimodal(                                      # cria instância de BotMultimodal
            token=token,
            modelo_yolo=args.model,
            confianca_minima=args.conf,
            idioma_reconhecimento=os.getenv("AUDIO_LANGUAGE", "pt-BR"),
        )

    print("\n" + "=" * 60)
    print(" BOT TELEGRAM COM INTELIGÊNCIA ARTIFICIAL EM EXECUÇÃO")
    print(f"• Classe ativa: {bot_instancia.__class__.__name__}")             # __class__.__name__ mostra o nome da classe usada
    print(f"• Token configurado: {bot_instancia.token_mascarado}")           # token_mascarado é um método de BotTelegram que retorna o token protegido
    print(f"• Modelo YOLO: {args.model}")
    print("• Pressione Ctrl+C para encerrar o bot a qualquer momento.")
    print("=" * 60 + "\n")

    try:
        bot_instancia.iniciar()                                              # conecta ao Telegram e fica escutando mensagens (loop contínuo).
    except KeyboardInterrupt:
        logger.info("Encerrando bot graciosamente...")                       # Captura a interrupção Ctrl+C.
    except Exception as erro:
        logger.exception("Falha crítica durante a execução do bot: %s", erro) # .exception captura erro inesperado, incluindo aviso e traceback
        sys.exit(1)


if __name__ == "__main__":
    main()



'''
--- VISUALIZAR FLUXO DE EXECUÇÃO: 

python bot.py --mode imagem --conf 0.4
        │
        ▼
 load_dotenv()  →  lê .env
        │
        ▼
 parse_argumentos()  →  lê o terminal
        │
        ├── --test-local? ──► roda tests/test_local.py e sai
        │
        ▼
 Valida o token ──► inválido? mostra ajuda e sai (código 1)
        │
        ▼
 Escolhe a classe (BotImagem / BotAudio / BotMultimodal)
        │
        ▼
 Mostra o banner e chama bot_instancia.iniciar()
        │
        ▼
 Fica rodando até Ctrl+C ou erro


'''




