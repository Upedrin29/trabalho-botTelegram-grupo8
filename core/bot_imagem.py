"""
Módulo de Visão Computacional do Bot Telegram.
Define a classe derivada BotImagem que herda de BotTelegram e implementa
o processamento e classificação de imagens com YOLO (ultralytics) e OpenCV.
"""

from __future__ import annotations
import io
import logging
import os
import time
from typing import Optional, Dict, Any, Tuple, List
import cv2
import numpy as np
from ultralytics import YOLO
from telegram import Update
from telegram.ext import ContextTypes

from .bot_base import BotTelegram

logger = logging.getLogger("BotImagem")


class BotImagem(BotTelegram):

    """
    Classe derivada de BotTelegram especializada no processamento de imagens.
    
    Aplica Polimorfismo ao sobrescrever o método 'processar_imagem' e integra
    o modelo YOLOv8 da Ultralytics e OpenCV para detecção, classificação
    e renderização visual de objetos.
    """

    def __init__(self,token: Optional[str] = None,modelo_yolo: str = "yolov8n.pt",confianca_minima: float = 0.25,**kwargs: Any):

        """
        Inicializa o Bot de Imagem carregando o modelo YOLO.
        
        :param token: Token de autenticação da API do Telegram (encapsulado na classe base).
        :param modelo_yolo: Caminho ou identificador do modelo YOLO (padrão: 'yolov8n.pt').
        :param confianca_minima: Limiar de confiança para considerar uma detecção válida (0.0 a 1.0).
        """

        super().__init__(token=token, **kwargs)

        self._modelo_yolo_nome = modelo_yolo or os.getenv("YOLO_MODEL", "yolov8n.pt")
        self._confianca_minima = confianca_minima
        
        logger.info("Carregando modelo YOLO (%s)...", self._modelo_yolo_nome)
        self._modelo = YOLO(self._modelo_yolo_nome)                                 # Carrega o modelo na memória, uma única vez, na criação do bot.

        logger.info("Modelo YOLO inicializado com sucesso.")

        # Paleta de cores para desenhar caixas de classes distintas no OpenCV
        self._cores_classes = self._gerar_paleta_cores(80)                         # Gera 80 cores, uma para cada classe do dataset COCO (o YOLOv8 padrão reconhece 80 classes: pessoa, carro, gato...).



    @staticmethod
    def _gerar_paleta_cores(num_cores: int = 80):

        """
        Gera uma paleta harmoniosa de cores BGR para distinguir visualmente classes no OpenCV.
        """

        np.random.seed(42)     # Semente fixa para cores consistentes entre execuções (a classe “pessoa” terá sempre a mesma cor).
        cores = []
        for _ in range(num_cores):
            b = int(np.random.randint(40, 240))        # Valores de 40 a 240 evitam cores muito escuras ou estouradas.
            g = int(np.random.randint(40, 240))
            r = int(np.random.randint(40, 240))
            cores.append((b, g, r))
        return cores



    def detectar_e_anotar(self, imagem_bgr: np.ndarray):

        """
        Executa a inferência com YOLO na imagem e anota os resultados usando OpenCV.
        
        :param imagem_bgr: Matriz NumPy da imagem em formato BGR (padrão OpenCV).
        :return: Tupla com (imagem_anotada, relatorio_metricas).
        """

        inicio = time.perf_counter()
        
        # Executa inferência YOLO analisa a imagem e devolve as detecções.
        resultados = self._modelo.predict(        
            source=imagem_bgr,
            conf=self._confianca_minima,
            verbose=False,
        )
        
        tempo_ms = (time.perf_counter() - inicio) * 1000.0

        imagem_anotada = imagem_bgr.copy()
        
        resultado = resultados[0]
        boxes = resultado.boxes
        nomes_classes = resultado.names
        
        contagem_por_classe: Dict[str, int] = {}
        detalhes_deteccoes: List[Dict[str, Any]] = []


        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())                              # o número da classe; .item() extrai o valor puro de um tensor.
                conf = float(box.conf[0].item())                             # a confiança (0 a 1).
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]            # coordenadas [x1, y1, x2, y2] (canto superior esquerdo e inferior direito).
                x1, y1, x2, y2 = map(int, coords)                            # converte para inteiros (pixels).
                
                nome_classe = nomes_classes.get(cls_id, f"classe_{cls_id}")
                contagem_por_classe[nome_classe] = contagem_por_classe.get(nome_classe, 0) + 1
                
                detalhes_deteccoes.append({
                    "classe": nome_classe,
                    "confianca": conf,
                    "bbox": (x1, y1, x2, y2),
                })
                
                # Renderização com OpenCV
                cor = self._cores_classes[cls_id % len(self._cores_classes)]
                self._desenhar_caixa_anotada(imagem_anotada, x1, y1, x2, y2, nome_classe, conf, cor)

        # Adiciona overlay de cabeçalho moderno na imagem via OpenCV
        self._desenhar_cabecalho_estatisticas(
            imagem_anotada,
            total_objetos=len(detalhes_deteccoes),
            tempo_ms=tempo_ms,
        )

        relatorio = {
            "total_objetos": len(detalhes_deteccoes),
            "tempo_inferencia_ms": round(tempo_ms, 2),
            "contagem_classes": contagem_por_classe,
            "deteccoes": detalhes_deteccoes,
            "resolucao": f"{imagem_bgr.shape[1]}x{imagem_bgr.shape[0]}",
        }


        return imagem_anotada, relatorio



    def _desenhar_caixa_anotada(self,img: np.ndarray,x1: int,y1: int,x2: int,y2: int,rotulo: str,confianca: float,cor: Tuple[int, int, int] ):

        """
        Desenha uma caixa delimitadora elegante com OpenCV contendo tarja de identificação.
        """

        # Caixa delimitadora
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2, cv2.LINE_AA)

        # Texto do rótulo
        texto = f"{rotulo} {confianca * 100:.1f}%"
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.55
        espessura = 1
        (largura_texto, altura_texto), linha_base = cv2.getTextSize(texto, fonte, escala, espessura)

        # Coordenadas do fundo do texto
        topo_fundo = max(y1 - altura_texto - 8, 0)
        baixo_fundo = y1
        direita_fundo = min(x1 + largura_texto + 8, img.shape[1])

        # Fundo sólido com a cor da classe
        cv2.rectangle(img, (x1, topo_fundo), (direita_fundo, baixo_fundo), cor, -1)
        
        # Texto em branco ou preto para contraste
        cor_texto = (255, 255, 255) if (cor[0] * 0.299 + cor[1] * 0.587 + cor[2] * 0.114) < 150 else (0, 0, 0)
        cv2.putText(
            img,
            texto,
            (x1 + 4, baixo_fundo - 4),
            fonte,
            escala,
            cor_texto,
            espessura,
            cv2.LINE_AA,
        )



    def _desenhar_cabecalho_estatisticas(
        self,
        img: np.ndarray,
        total_objetos: int,
        tempo_ms: float,
    ) -> None:
        """
        Desenha uma barra de status translúcida no topo da imagem com resumo das métricas.
        """
        altura_barra = 36
        h, w = img.shape[:2]
        if h < 100 or w < 200:
            return

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, altura_barra), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        texto_status = f"YOLOv8 | Objetos: {total_objetos} | Tempo: {tempo_ms:.1f}ms"
        cv2.putText(
            img,
            texto_status,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )




    # -------------------------------------------------------------------------
    # Implementação Polimórfica: processar_imagem
    # -------------------------------------------------------------------------




    async def processar_imagem(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        """
        Sobrescreve polimorficamente o método 'processar_imagem' da classe base.
        Baixa a foto recebida pelo Telegram, processa via YOLO + OpenCV e envia a resposta.
        """
        
        msg = update.message
        if not msg:
            return

        # Notifica o usuário que o processamento começou
        status_msg = await msg.reply_text("🔍 *Analisando imagem com YOLOv8...*", parse_mode="Markdown")

        try:
            # Obtém a foto na maior resolução disponível
            if msg.photo:
                foto = msg.photo[-1]
            elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
                foto = msg.document
            else:
                await status_msg.edit_text("Nenhuma imagem válida detectada no arquivo.")
                return

            arquivo_telegram = await context.bot.get_file(foto.file_id)
            buffer_bytes = io.BytesIO()
            await arquivo_telegram.download_to_memory(buffer_bytes)
            buffer_bytes.seek(0)


            # Decodifica imagem para matriz OpenCV
            array_np = np.frombuffer(buffer_bytes.getvalue(), dtype=np.uint8)        # frombuffer: transforma os bytes em um array NumPy de inteiros de 8 bits.
            imagem_bgr = cv2.imdecode(array_np, cv2.IMREAD_COLOR)                    # imdecode: decodifica esses bytes (JPEG, PNG...) em uma matriz de pixels colorida (BGR). Devolve None se falhar, por isso o teste seguinte.

 
            if imagem_bgr is None:
                await status_msg.edit_text("Falha ao decodificar os pixels da imagem.")
                return

            # Executa inferência e anotação
            imagem_anotada, relatorio = self.detectar_e_anotar(imagem_bgr)

            # Converte a imagem anotada de volta para JPEG
            sucesso, buffer_codificado = cv2.imencode(".jpg", imagem_anotada, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            if not sucesso:
                await status_msg.edit_text("Erro ao codificar imagem resultante.")
                return

            foto_para_envio = io.BytesIO(buffer_codificado.tobytes())        # Cria um arquivo virtual e dá um nome a ele (o Telegram precisa disso para reconhecer o formato).
            foto_para_envio.name = "deteccao_yolo.jpg"

            # Formata o relatório textual
            legenda = self._formatar_relatorio_markdown(relatorio)

            # Deleta a mensagem de status temporária
            await status_msg.delete()

            # Envia a foto anotada com a legenda gerado em markdown
            await msg.reply_photo(
                photo=foto_para_envio,
                caption=legenda,
                parse_mode="Markdown",
            )


        except Exception as erro:
            logger.exception("Erro durante o processamento da imagem: %s", erro)
            await status_msg.edit_text(f"Ocorreu um erro ao processar a imagem:\n`{str(erro)}`", parse_mode="Markdown")




    def _formatar_relatorio_markdown(self, relatorio: Dict[str, Any]):

        """
        Gera um resumo formatado em Markdown com contagens e métricas da inferência.
        """

        total = relatorio["total_objetos"]
        tempo = relatorio["tempo_inferencia_ms"]
        classes = relatorio["contagem_classes"]

        if total == 0:
            return (
                f"*Resultado da Análise Visual (YOLOv8)*\n\n"
                f"ℹ Nenhum objeto com confiança acima de `{self._confianca_minima * 100:.0f}%` "
                f"foi detectado na imagem.\n"
                f" Tempo de inferência: `{tempo:.1f}ms`"
            )

        linhas = [
            f"*Classificação e Detecção de Imagem (YOLOv8)*",
            f"• **Total de Objetos Detectados:** `{total}`",
            f"• **Tempo de Inferência:** `{tempo:.1f}ms`",
            f"• **Resolução:** `{relatorio['resolucao']}`\n",
            "*Classes Identificadas:*",
        ]

        for classe, qtd in sorted(classes.items(), key=lambda item: item[1], reverse=True):
            # Obtém a maior confiança dessa classe
            maior_conf = max(
                (d["confianca"] for d in relatorio["deteccoes"] if d["classe"] == classe),
                default=0.0,
            )
            linhas.append(f"  ▫️ **{classe.capitalize()}**: `{qtd}` item(ns) (Confiança máx: `{maior_conf * 100:.1f}%`)")

        return "\n".join(linhas)





    def processar_imagem_local(self, caminho_arquivo: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Método utilitário para testes locais offline.
        Lê uma imagem do sistema de arquivos e executa a detecção.
        """
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
        
        img = cv2.imread(caminho_arquivo)
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {caminho_arquivo}")
        
        return self.detectar_e_anotar(img)





'''
Usuário envia uma foto
        │
        ▼
despachar_mensagem() → tipo = "imagem"
        │
        ▼
processar_imagem()  ◄── versão do BotImagem (polimorfismo)
        ├─ responde "Analisando..."
        ├─ baixa a maior resolução → bytes
        ├─ cv2.imdecode → matriz BGR
        ▼
detectar_e_anotar()
        ├─ YOLO.predict() → caixas, classes, confianças
        ├─ _desenhar_caixa_anotada() (cada objeto)
        └─ _desenhar_cabecalho_estatisticas()
        ▼
cv2.imencode → JPEG
        ▼
_formatar_relatorio_markdown()
        ▼
apaga o status e envia foto + legenda

'''