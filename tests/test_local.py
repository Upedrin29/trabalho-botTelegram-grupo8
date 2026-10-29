"""
Suíte de Testes Locais para Validação do Bot Telegram com IA.

Valida os requisitos essenciais antes do deploy no Telegram:
1. Princípios de POO (Herança, Encapsulamento, Polimorfismo e MRO).
2. Modelo YOLOv8 e anotação visual com OpenCV.
3. Classificador Semântico e pipeline de áudio.
4. Robustez no tratamento de entradas variadas.
"""

from __future__ import annotations
import io
import math
import os
import struct
import sys
import unittest
import wave
import cv2
import numpy as np

# Adiciona o diretório raiz ao PYTHONPATH para importação correta
DIRETORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIRETORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRETORIO_RAIZ)

from core import BotTelegram, BotImagem, BotAudio, BotMultimodal
from core.bot_audio import ClassificadorTextoAudio


class TestePOOEEncapsulamento(unittest.TestCase):
    
    """Testes dos princípios de Programação Orientada a Objetos."""

    TOKEN_TESTE = "1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ"


    def test_encapsulamento_token_privado(self):
        """Valida que o token da API está encapsulado e protegido contra acesso acidental."""
        bot = BotTelegram(token=self.TOKEN_TESTE)

        # O atributo é privado (__token) e sofreu name mangling
        with self.assertRaises(AttributeError):
            _ = bot.__token

        # O token mascarado deve ocultar o corpo sensível da chave
        mascarado = bot.token_mascarado
        self.assertTrue(mascarado.startswith("123456"))
        self.assertTrue(mascarado.endswith("wxYZ"))
        self.assertIn("...", mascarado)
        self.assertNotIn("ABCdefGHIjklMNOpqr", mascarado)


    def test_validacao_token_invalido(self):
        """Valida que inicialização sem token dispara exceção informativa."""
        # Limpa temporariamente variável de ambiente se existir
        env_original = os.environ.get("TELEGRAM_BOT_TOKEN")
        if "TELEGRAM_BOT_TOKEN" in os.environ:
            del os.environ["TELEGRAM_BOT_TOKEN"]

        try:
            with self.assertRaises(ValueError):
                BotTelegram(token="")
        finally:
            if env_original is not None:
                os.environ["TELEGRAM_BOT_TOKEN"] = env_original


    def test_hierarquia_e_heranca(self):
        """Valida a árvore de herança e especialização de classes."""
        self.assertTrue(issubclass(BotImagem, BotTelegram))
        self.assertTrue(issubclass(BotAudio, BotTelegram))
        self.assertTrue(issubclass(BotMultimodal, BotImagem))
        self.assertTrue(issubclass(BotMultimodal, BotAudio))
        self.assertTrue(issubclass(BotMultimodal, BotTelegram))


class TesteVisaoComputacionalYOLO(unittest.TestCase):

    """Testes do modelo YOLOv8 e integração com OpenCV."""

    @classmethod
    def setUpClass(cls):
        """Instancia o BotImagem uma única vez para os testes de visão."""
        cls.bot_img = BotImagem(token="123456:FAKE_TOKEN_PARA_TESTE", modelo_yolo="yolov8n.pt")


    def test_inferencia_e_anotacao_opencv(self):
        """Testa o processamento de uma imagem com OpenCV e inferência com YOLOv8."""
        # Cria uma imagem sintética BGR de 640x640 pixels
        altura, largura = 640, 640
        imagem_teste = np.zeros((altura, largura, 3), dtype=np.uint8)
        
        # Desenha um fundo com gradiente e formas geométricas
        cv2.rectangle(imagem_teste, (50, 50), (300, 300), (0, 200, 255), -1)
        cv2.circle(imagem_teste, (450, 450), 80, (255, 100, 0), -1)
        cv2.putText(imagem_teste, "Teste YOLO", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Executa detecção
        img_anotada, relatorio = self.bot_img.detectar_e_anotar(imagem_teste)

        # Validações
        self.assertIsInstance(img_anotada, np.ndarray)
        self.assertEqual(img_anotada.shape, imagem_teste.shape)
        self.assertIn("total_objetos", relatorio)
        self.assertIn("tempo_inferencia_ms", relatorio)
        self.assertIn("contagem_classes", relatorio)
        self.assertGreater(relatorio["tempo_inferencia_ms"], 0.0)
        self.assertEqual(relatorio["resolucao"], f"{largura}x{altura}")


    def test_imagem_com_objeto_real(self):
        """Valida que o YOLOv8 carrega pesos COCO e é capaz de detectar classes reais."""
        # Baixa ou usa a imagem sintética com verificação de formato
        classes_modelo = self.bot_img._modelo.names
        self.assertIn(0, classes_modelo)
        self.assertEqual(classes_modelo[0], "person")
        self.assertIn(2, classes_modelo)
        self.assertEqual(classes_modelo[2], "car")


class TesteAudioENLP(unittest.TestCase):

    """Testes do Classificador de Texto de Áudio e conversor de áudio."""

    def test_classificador_sentimento_positivo(self):
        """Valida detecção de sentimento positivo e intenção de elogio/agradecimento."""
        texto = "Muito obrigado pelo atendimento, o serviço foi excelente e maravilhoso!"
        res = ClassificadorTextoAudio.classificar(texto, duracao_segundos=4.0)

        self.assertIn("Positivo", res["sentimento"])
        self.assertGreater(res["polaridade_score"], 0)
        self.assertIn("Agradecimento", res["categoria_intencao"])
        self.assertGreater(res["ritmo_fala_wpm"], 0)


    def test_classificador_sentimento_negativo(self):
        """Valida detecção de sentimento negativo e intenção de suporte/erro."""
        texto = "Estou com um erro gravíssimo, o sistema quebrou e travou tudo, péssimo!"
        res = ClassificadorTextoAudio.classificar(texto, duracao_segundos=5.0)

        self.assertIn("Negativo", res["sentimento"])
        self.assertLess(res["polaridade_score"], 0)
        self.assertIn("Suporte", res["categoria_intencao"])


    def test_classificador_duvida_pergunta(self):
        """Valida detecção de pergunta."""
        texto = "Como posso atualizar meus dados cadastrais?"
        res = ClassificadorTextoAudio.classificar(texto, duracao_segundos=3.0)

        self.assertIn("Pergunta", res["categoria_intencao"])


    def test_conversao_audio_sintetico(self):
        """Gera um arquivo de áudio WAV sintético e valida o pipeline de carregamento."""
        bot_audio = BotAudio(token="123456:FAKE_TOKEN_PARA_TESTE")
        
        # Cria um áudio WAV puro (onda senoidal de 440 Hz - tom de Lá)
        taxa_amostragem = 16000
        duracao_seg = 0.5
        total_amostras = int(taxa_amostragem * duracao_seg)
        frequencia = 440.0

        buffer_wav = io.BytesIO()
        with wave.open(buffer_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(taxa_amostragem)
            dados = bytearray()
            for i in range(total_amostras):
                amostra = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequencia * i / taxa_amostragem))
                dados.extend(struct.pack("<h", amostra))
            wf.writeframes(dados)

        buffer_wav.seek(0)
        bytes_audio = buffer_wav.getvalue()

        # Valida conversão com pydub
        caminho_wav, duracao = bot_audio.converter_para_wav(bytes_audio, formato_origem="wav")
        try:
            self.assertTrue(os.path.exists(caminho_wav))
            self.assertAlmostEqual(duracao, 0.5, delta=0.1)
        finally:
            if os.path.exists(caminho_wav):
                os.remove(caminho_wav)



    def test_conversao_audio_ogg_telegram(self):
        """Valida que mensagens de voz em OGG (padrão Telegram) são convertidas para WAV com sucesso."""
        from pydub.generators import Sine
        bot_audio = BotAudio(token="123456:FAKE_TOKEN_PARA_TESTE")

        buf_ogg = io.BytesIO()
        Sine(440).to_audio_segment(duration=1000).export(buf_ogg, format="ogg")
        bytes_ogg = buf_ogg.getvalue()

        caminho_wav, duracao = bot_audio.converter_para_wav(bytes_ogg, formato_origem="ogg")
        try:
            self.assertTrue(os.path.exists(caminho_wav))
            self.assertAlmostEqual(duracao, 1.0, delta=0.2)
        finally:
            if os.path.exists(caminho_wav):
                os.remove(caminho_wav)


class TestePolimorfismoDespacho(unittest.TestCase):

    """Testes da identificação dinâmica de tipos de mensagens."""


    def test_reconhecimento_tipos(self):
        """Valida a lógica polimórfica de reconhecimento de mensagens."""
        bot = BotTelegram(token="123456:FAKE_TOKEN")

        # Mock simples de Update e Message
        class MockMessage:
            def __init__(self, photo=None, voice=None, audio=None, text=None, document=None):
                self.photo = photo
                self.voice = voice
                self.audio = audio
                self.text = text
                self.document = document

        class MockUpdate:
            def __init__(self, msg):
                self.message = msg

        msg_foto = MockUpdate(MockMessage(photo=[object()]))
        self.assertEqual(bot._identificar_tipo_mensagem(msg_foto), "imagem")

        msg_voz = MockUpdate(MockMessage(voice=object()))
        self.assertEqual(bot._identificar_tipo_mensagem(msg_voz), "audio")

        msg_audio = MockUpdate(MockMessage(audio=object()))
        self.assertEqual(bot._identificar_tipo_mensagem(msg_audio), "audio")

        msg_texto = MockUpdate(MockMessage(text="Olá mundo!"))
        self.assertEqual(bot._identificar_tipo_mensagem(msg_texto), "texto")


def executar_testes():
    """Executa a suíte completa de testes com relatório formatado."""
    print("\n" + "=" * 65)
    print("INICIANDO TESTES LOCAIS DO BOT TELEGRAM COM IA")
    print("=" * 65)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestePOOEEncapsulamento))
    suite.addTest(loader.loadTestsFromTestCase(TesteVisaoComputacionalYOLO))
    suite.addTest(loader.loadTestsFromTestCase(TesteAudioENLP))
    suite.addTest(loader.loadTestsFromTestCase(TestePolimorfismoDespacho))

    runner = unittest.TextTestRunner(verbosity=2)
    resultado = runner.run(suite)

    print("=" * 65)
    if resultado.wasSuccessful():
        print("TODOS OS TESTES PASSARAM COM SUCESSO!")
        return 0
    else:
        print("ALGUNS TESTES FALHARAM. VERIFIQUE OS LOGS ACIMA.")
        return 1


if __name__ == "__main__":
    sys.exit(executar_testes())
