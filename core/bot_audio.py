"""
Módulo de Processamento de Áudio e Fala do Bot Telegram.
Define a classe derivada BotAudio que herda de BotTelegram e implementa
conversão de áudio, Speech-to-Text e classificação semântica/sentimento do conteúdo.
"""

from __future__ import annotations
import asyncio
import io
import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from typing import Optional, Dict, Any, Tuple
import imageio_ffmpeg
from pydub import AudioSegment
import speech_recognition as sr
from telegram import Update
from telegram.ext import ContextTypes

from .bot_base import BotTelegram

# Configura o executável do ffmpeg para o pydub e para o processo
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG_EXE

# Garante que o diretório do FFmpeg esteja no PATH do sistema
_ffmpeg_dir = os.path.dirname(FFMPEG_EXE)
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


logger = logging.getLogger("BotAudio")



class ClassificadorTextoAudio:

    """
    Componente de Inteligência Artificial para análise semântica,
    classificação de sentimento e categorização de intenções do texto falado (definida manualmente).
    """

    # Léxico em português para análise de sentimento e tonalidade
    PALAVRAS_POSITIVAS = {
        "bom", "boa", "otimo", "ótimo", "excelente", "maravilha", "maravilhoso", "legal",
        "parabens", "parabéns", "obrigado", "obrigada", "valeu", "adorei", "amei", "gostei",
        "perfeito", "sucesso", "positivo", "funciona", "top", "sensacional", "bacana", "ajudou"
    }

    PALAVRAS_NEGATIVAS = {
        "ruim", "pessimo", "péssimo", "horrivel", "horrível", "falha", "erro", "defeito",
        "quebrou", "nao", "não", "nunca", "lento", "travou", "odiei", "inutil", "inútil",
        "dificil", "difícil", "chato", "reclamacao", "reclamação", "problema", "bosta", "merda"
    }

    # Padrões para classificação de intenção
    PADROES_INTENCAO = {
        "Pergunta / Dúvida": [
            r"\b(como|quando|onde|porque|por que|por quê|qual|quem|quanto|será|sera)\b",
            r"\?$",
        ],
        "Suporte / Ajuda": [
            r"\b(ajuda|socorro|suporte|erro|bug|travou|falhou|consertar|resolver|ajudar|problema)\b",
        ],
        "Saudação": [
            r"\b(ola|olá|oi|opa|e ai|e aí|bom dia|boa tarde|boa noite|tudo bem|fala)\b",
        ],
        "Comando / Solicitação": [
            r"\b(faça|faca|execute|envie|mande|mostre|abra|pare|inicie|gere|crie|calcule)\b",
        ],
        "Agradecimento / Feedback": [
            r"\b(obrigado|obrigada|valeu|grato|agradeço|agradeco|parabens|parabéns)\b",
        ],
    }

    @classmethod
    def classificar(cls, texto: str, duracao_segundos: float = 0.0):

        """
        Classifica o texto transcrito em sentimento, categoria de intenção e extrai métricas linguísticas.
        
        :param texto: Texto transcrito do áudio.
        :param duracao_segundos: Duração do áudio original em segundos.
        :return: Dicionário detalhado com as classificações e pontuações.
        """

        if not texto or not texto.strip():
            return {
                "sentimento": "Indeterminado",
                "polaridade_score": 0.0,
                "categoria_intencao": "Inaudível / Vazio",
                "total_palavras": 0,
                "palavras_chave": [],
                "ritmo_fala_wpm": 0.0,
            }

        texto_limpo = texto.lower()
        palavras = re.findall(r"\b[a-zá-úãõç]+\b", texto_limpo)
        total_palavras = len(palavras)


        # 1. Análise de Sentimento
        positivas_encontradas = [p for p in palavras if p in cls.PALAVRAS_POSITIVAS]
        negativas_encontradas = [p for p in palavras if p in cls.PALAVRAS_NEGATIVAS]

        score_sentimento = len(positivas_encontradas) - len(negativas_encontradas)
        if score_sentimento > 0:
            sentimento = "Positivo"
        elif score_sentimento < 0:
            sentimento = "Negativo"
        else:
            sentimento = "Neutro / Informativo"


        # 2. Classificação de Intenção
        intencoes_detectadas = []
        for intencao, padroes in cls.PADROES_INTENCAO.items():
            for padrao in padroes:
                if re.search(padrao, texto_limpo):
                    intencoes_detectadas.append(intencao)
                    break

        categoria_principal = intencoes_detectadas[0] if intencoes_detectadas else "Declaração Geral / Outro 💬"


        # 3. Métricas de Fala (WPM - Words Per Minute)
        wpm = 0.0
        if duracao_segundos > 0:
            wpm = round((total_palavras / duracao_segundos) * 60, 1)



        return {
            "sentimento": sentimento,
            "polaridade_score": score_sentimento,
            "termos_positivos": positivas_encontradas,
            "termos_negativos": negativas_encontradas,
            "categoria_intencao": categoria_principal,
            "total_palavras": total_palavras,
            "ritmo_fala_wpm": wpm,
        }




class BotAudio(BotTelegram):

    """
    Classe derivada de BotTelegram especializada no processamento de áudios.
    
    Aplica Polimorfismo ao sobrescrever o método 'processar_audio', convertendo
    formatos OGG/Opus do Telegram para WAV e executando Speech-to-Text seguido
    de classificação de sentimento e intenção.
    """

    def __init__( self,token: Optional[str] = None, idioma_reconhecimento: str = "pt-BR", **kwargs: Any,):

        """
        Inicializa o Bot de Áudio.
        
        :param token: Token de autenticação da API do Telegram.
        :param idioma_reconhecimento: Código BCP-47 do idioma (padrão: 'pt-BR').
        """

        super().__init__(token=token, **kwargs)

        self._idioma = idioma_reconhecimento or os.getenv("AUDIO_LANGUAGE", "pt-BR")
        self._recognizer = sr.Recognizer()
        
        # Ajustes de sensibilidade do reconhecedor de fala (mais util quando microfone ao vivo)
        self._recognizer.energy_threshold = 300
        self._recognizer.dynamic_energy_threshold = True



    def converter_para_wav(self, bytes_audio: bytes, formato_origem: str = "ogg"):

        """
        Converte dados de áudio em bytes (ex: OGG Opus do Telegram) para um arquivo WAV temporário.
        Utiliza o executável FFmpeg diretamente para converter qualquer codec/contêiner para WAV mono 16kHz,
        eliminando a dependência do ffprobe no Windows e garantindo compatibilidade total com áudios do Telegram.
        
        :param bytes_audio: Conteúdo binário do áudio.
        :param formato_origem: Extensão do formato de origem (ex: 'ogg', 'mp3', 'm4a').
        :return: Tupla com (caminho_arquivo_wav, duracao_segundos).
        """

        caminho_in = None
        caminho_wav = None
        try:
            # Salva os bytes recebidos em um arquivo temporário com a extensão de origem
            ext = (formato_origem or "ogg").lstrip(".").lower()
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f_in:
                f_in.write(bytes_audio)
                caminho_in = f_in.name

            # Cria arquivo temporário de destino para o WAV
            arquivo_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            caminho_wav = arquivo_wav.name
            arquivo_wav.close()

            # Executa conversão via FFmpeg diretamente: 1 canal (mono), taxa 16kHz
            cmd = [
                FFMPEG_EXE,
                "-y",
                "-i", caminho_in,
                "-ac", "1",
                "-ar", "16000",
                caminho_wav,
            ]
            resultado_proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if resultado_proc.returncode != 0:
                erro_detalhe = resultado_proc.stderr.decode("utf-8", errors="ignore")
                logger.error("FFmpeg falhou ao converter áudio: %s", erro_detalhe)
                raise RuntimeError(f"Falha na conversão de áudio via FFmpeg: {erro_detalhe[-200:]}")

            # Lê a duração exata do WAV gerado através do módulo nativo wave
            with wave.open(caminho_wav, "rb") as wf:
                duracao_seg = float(wf.getnframes()) / float(wf.getframerate())

            return caminho_wav, duracao_seg

        except Exception as e:
            if caminho_wav and os.path.exists(caminho_wav):
                try:
                    os.remove(caminho_wav)
                except OSError:
                    pass
            raise e

        finally:
            if caminho_in and os.path.exists(caminho_in):
                try:
                    os.remove(caminho_in)
                except OSError:
                    pass



    def transcrever_audio_wav(self, caminho_wav: str):

        """
        Executa o reconhecimento de fala no arquivo WAV utilizando SpeechRecognition.
        
        :param caminho_wav: Caminho absoluto para o arquivo de áudio WAV.
        :return: Texto transcrito.
        """

        with sr.AudioFile(caminho_wav) as fonte_audio:
            # Captura o áudio
            dados_audio = self._recognizer.record(fonte_audio)
            
            # Reconhecimento via Google Speech Recognition API (gratuito e nativo em pt-BR)
            texto_transcrito = self._recognizer.recognize_google(                           # recognize_google envia o áudio para a API de reconhecimento do Google e devolve o texto.
                dados_audio,
                language=self._idioma,
            )
            return texto_transcrito



    def processar_audio_completo(self, bytes_audio: bytes, formato_origem: str = "ogg"):

        """
        Pipeline completo: conversão para WAV -> Speech-to-Text -> Classificação com IA.
        
        :param bytes_audio: Bytes brutos do áudio recebido.
        :param formato_origem: Formato do arquivo (ogg, mp3, etc.).
        :return: Dicionário com transcrição, duração e relatórios de classificação.
        """

        caminho_wav = None
        try:
            inicio = time.perf_counter()
            caminho_wav, duracao_seg = self.converter_para_wav(bytes_audio, formato_origem)

            try:
                transcricao = self.transcrever_audio_wav(caminho_wav)
            except sr.UnknownValueError:
                transcricao = ""
            except sr.RequestError as e:
                raise ConnectionError(f"Erro no serviço de reconhecimento de fala: {e}")

            tempo_proc = (time.perf_counter() - inicio) * 1000.0

            # Classifica o conteúdo transcrito
            classificacao = ClassificadorTextoAudio.classificar(transcricao, duracao_seg)

            return {
                "sucesso": bool(transcricao),
                "transcricao": transcricao,
                "duracao_segundos": round(duracao_seg, 2),
                "tempo_processamento_ms": round(tempo_proc, 2),
                "classificacao": classificacao,
            }

        finally:
            if caminho_wav and os.path.exists(caminho_wav):
                try:
                    os.remove(caminho_wav)
                except OSError:
                    pass




    # -------------------------------------------------------------------------
    # Implementação Polimórfica: processar_audio
    # -------------------------------------------------------------------------




    async def processar_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        """
        Sobrescreve polimorficamente o método 'processar_audio' da classe base.
        Baixa a mensagem de voz ou arquivo de áudio do Telegram, transcreve e classifica o conteúdo.
        """
        
        msg = update.message
        if not msg:
            return

        status_msg = await msg.reply_text("🎙️ *Transcrevendo e classificando o áudio...*", parse_mode="Markdown")

        try:
            formato = "ogg"
            if msg.voice:
                arquivo_info = msg.voice
                formato = "ogg"
            elif msg.audio:
                arquivo_info = msg.audio
                if arquivo_info.mime_type and "mp3" in arquivo_info.mime_type:
                    formato = "mp3"
                elif arquivo_info.mime_type and "wav" in arquivo_info.mime_type:
                    formato = "wav"
                elif arquivo_info.mime_type and "m4a" in arquivo_info.mime_type:
                    formato = "m4a"
                else:
                    formato = "ogg"
            elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/"):
                arquivo_info = msg.document
                if arquivo_info.file_name and "." in arquivo_info.file_name:
                    formato = arquivo_info.file_name.rsplit(".", 1)[-1].lower()
                elif "mp3" in arquivo_info.mime_type:
                    formato = "mp3"
                elif "wav" in arquivo_info.mime_type:
                    formato = "wav"
                else:
                    formato = "ogg"
            else:
                await status_msg.edit_text("❌ Nenhum payload de áudio válido encontrado.")
                return


            arquivo_telegram = await context.bot.get_file(arquivo_info.file_id)      # Pede ao Telegram os dados do arquivo pelo file_id
            buffer_bytes = io.BytesIO()
            await arquivo_telegram.download_to_memory(buffer_bytes)                  # Baixa direto para a memória (sem salvar em disco).
            bytes_audio = buffer_bytes.getvalue()                                    # Pega o conteúdo binário bruto.



            # Inicia o pipeline de processamento em thread separada para não bloquear o loop assíncrono
            # O bot continua responsivo para outros comandos enquanto a fala é transcrita.
            resultado = await asyncio.to_thread(self.processar_audio_completo, bytes_audio, formato)


            # Verifica se a transcrição foi bem-sucedida
            if not resultado["sucesso"]:
                await status_msg.edit_text(
                    "⚠️ *Não foi possível transcrever a fala com clareza.*\n\n"
                    "Dicas:\n"
                    "• Tente falar um pouco mais perto do microfone;\n"
                    "• Evite ruídos de fundo muito intensos;\n"
                    "• Duração do áudio recebido: `{dur:.1f}s`".format(dur=resultado["duracao_segundos"]),
                    parse_mode="Markdown",
                )
                return

            resposta_md = self._formatar_relatorio_audio_markdown(resultado)
            try:
                await status_msg.edit_text(resposta_md, parse_mode="Markdown")
            except Exception:
                # Fallback caso a transcrição contenha caracteres que interfiram no Markdown
                await status_msg.edit_text(resposta_md)


        except Exception as erro:
            logger.exception("Erro durante o processamento de áudio: %s", erro)
            await status_msg.edit_text(f"❌ Ocorreu um erro ao processar o áudio:\n{str(erro)}")



    def _formatar_relatorio_audio_markdown(self, resultado: Dict[str, Any]):

        """
        Gera relatório estilizado em Markdown com transcrição e classificações do áudio.
        """

        transcricao = resultado["transcricao"]
        duracao = resultado["duracao_segundos"]
        tempo_ms = resultado["tempo_processamento_ms"]
        cls_info = resultado["classificacao"]

        linhas = [
            " *Transcrição e Classificação de Áudio*",
            "",
            f" *Transcrição do Áudio:*",
            f"> \"_{transcricao}_\"",
            "",
            " *Classificação Semântica com IA:*",
            f"• **Sentimento:** `{cls_info['sentimento']}`",
            f"• **Intenção Identificada:** `{cls_info['categoria_intencao']}`",
            "",
            " *Métricas Fonéticas e do Áudio:*",
            f"• **Duração:** `{duracao:.1f} segundos`",
            f"• **Total de Palavras:** `{cls_info['total_palavras']}`",
            f"• **Velocidade de Fala:** `{cls_info['ritmo_fala_wpm']} palavras/min`",
            f"• **Tempo de Processamento:** `{tempo_ms:.1f}ms`",
        ]

        if cls_info.get("termos_positivos"):
            linhas.append(f"• **Termos Positivos:** `{', '.join(cls_info['termos_positivos'])}`")
        if cls_info.get("termos_negativos"):
            linhas.append(f"• **Termos Negativos:** `{', '.join(cls_info['termos_negativos'])}`")

        return "\n".join(linhas)




    def processar_audio_local(self, caminho_arquivo: str):

        """
        Método utilitário para testes locais offline.
        Lê um arquivo de áudio do sistema e executa o pipeline.
        """

        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho_arquivo}")

        _, extensao = os.path.splitext(caminho_arquivo)
        formato = extensao.lstrip(".").lower() or "wav"

        with open(caminho_arquivo, "rb") as f:
            bytes_audio = f.read()

        return self.processar_audio_completo(bytes_audio, formato_origem=formato)




'''
Usuário envia mensagem de voz
        │
        ▼
despachar_mensagem() (BotTelegram) → tipo = "audio"
        │
        ▼
processar_audio()  ◄── versão do BotAudio (polimorfismo)
        ├─ responde "Transcrevendo..."
        ├─ baixa o áudio para a memória
        ▼
processar_audio_completo()
        ├─ converter_para_wav()   → OGG → WAV mono 16kHz
        ├─ transcrever_audio_wav() → Google Speech API
        └─ ClassificadorTextoAudio.classificar()
        ▼
_formatar_relatorio_audio_markdown()
        ▼
edita a mensagem com o relatório final

'''