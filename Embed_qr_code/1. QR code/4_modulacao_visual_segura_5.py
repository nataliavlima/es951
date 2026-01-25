import os
import cv2
import numpy as np
import qrcode
import qrcode.constants
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURAÇÃO
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA = "https://www.fem.unicamp.br/"
QR_VERSION = 10
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H
MODULE_SIZE = 10

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR, "imagens_base", 
    #"bob.png"
    #"FERB.jpg" 
    "homer.png" 
    #"dogs.jpeg" 
    #"donald.jpg"
    #"coragem.jpg"
    #"picapau.jpg"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_QR = os.path.join(OUTPUT_DIR, "qr_modulado_perceptual_HOMER.png")
OUTPUT_DEBUG = os.path.join(OUTPUT_DIR, "mapa_perceptual.png")

# ======================================================
# ETAPA 1 — GERAR MATRIZ LÓGICA DO QR
# ======================================================

def generate_qr_matrix(data, version, error_correction):
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=1,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=False)
    # qrcode retorna True para módulo PRETO
    return np.array(qr.get_matrix(), dtype=np.uint8)

# ======================================================
# UTIL — CROP QUADRADO CENTRAL
# ======================================================

def center_crop_square(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0+side, x0:x0+side]

# ======================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO DA IMAGEM BASE (COLORIDA + MAPA EM CINZA)
# ======================================================

def preprocess_base_image_color(image_path, target_shape):
    """
    Retorna:
      - base_bgr_resized: imagem BGR (0..255) redimensionada para (H,W) do QR
      - base_gray_float: mapa em cinza float (0..1) do mesmo tamanho
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Imagem base nao encontrada.")

    img_bgr = center_crop_square(img_bgr)
    # target_shape = (H, W) em módulos
    H, W = target_shape
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # equalização leve para destacar conteúdo (ajuda a percepção)
    gray_eq = cv2.equalizeHist(gray)

    base_gray_float = gray_eq.astype(np.float32) / 255.0
    return img_bgr, base_gray_float

# ======================================================
# MELHORIA PERCEPTUAL — CONTORNOS + MAPA
# ======================================================

def edge_enhanced_map(base_gray_float):
    edges = cv2.Canny(
        (base_gray_float * 255).astype(np.uint8),
        threshold1=60,
        threshold2=140
    )
    return edges.astype(np.float32) / 255.0

def perceptual_map(base_gray_float, edge_map, gamma=0.65):
    """
    gamma mais alto -> dá mais peso ao conteúdo base
    """
    combined = gamma * base_gray_float + (1.0 - gamma) * edge_map
    return np.clip(combined, 0.0, 1.0)

# ======================================================
# NOVO — MAPA DE TEXTURA (para alpha adaptativo)
# ======================================================

def texture_map(base_gray_float, ksize=3):
    """
    Usa magnitude do gradiente como proxy de textura (0..1).
    """
    img = (base_gray_float * 255).astype(np.uint8)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)

    # normaliza robusto (evita estourar por poucos pixels)
    p95 = np.percentile(mag, 95)
    if p95 <= 1e-6:
        return np.zeros_like(base_gray_float, dtype=np.float32)

    t = np.clip(mag / p95, 0.0, 1.0).astype(np.float32)
    return t

# ======================================================
# ETAPA 4 — RENDERIZAÇÃO HSV (S e V) + ALPHA ADAPTATIVO
# ======================================================

def render_modulated_qr_hsv(
    qr_matrix,
    base_bgr_small,     # (H,W,3) 0..255
    perceptual_small,   # (H,W) 0..1
    texture_small,      # (H,W) 0..1
    module_size,
    # parâmetros de controle
    alpha_min=0.32,     # QR pesa mais em regiões lisas
    alpha_max=0.72,     # QR pesa menos em regiões texturizadas
    v_black=15,         # alvo de V para módulos pretos (0..255) - mais baixo = mais contraste
    v_white=245,        # alvo de V para módulos brancos
    s_boost=1.20        # boost de saturação do base (ajuda a "aparecer")
):
    """
    Ideia:
      - Repetimos a imagem base (pixel por módulo) para o tamanho final.
      - No HSV: aumentamos S do base (s_boost).
      - Para cada módulo, empurramos V em direção a v_black ou v_white
        com um alpha que depende da textura.
      - Usamos perceptual_small para reforçar um pouco a presença do conteúdo.
    """
    H, W = qr_matrix.shape

    # base HSV em baixa resolução (H,W)
    base_hsv = cv2.cvtColor(base_bgr_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    Hch, Sch, Vch = cv2.split(base_hsv)

    # boost de saturação (limitado)
    Sch = np.clip(Sch * s_boost, 0, 255)

    # alpha adaptativo: mais textura -> alpha menor (mais base aparece)
    # alpha in [alpha_min, alpha_max] invertendo pela textura
    alpha = alpha_max - (alpha_max - alpha_min) * texture_small
    alpha = alpha.astype(np.float32)

    # reforço leve guiado por perceptual (mais perceptual -> alpha um pouco menor)
    # para deixar a imagem aparecer mais onde ela é "boa"
    alpha = np.clip(alpha * (1.0 - 0.18 * perceptual_small), 0.05, 0.95)

    # alvo de V por módulo (preto/branco)
    # qr_matrix == 1 -> preto
    targetV = np.where(qr_matrix == 1, v_black, v_white).astype(np.float32)

    # mistura em V: V_final = (1-alpha)*V_base + alpha*V_target
    V_final = (1.0 - alpha) * Vch + alpha * targetV
    V_final = np.clip(V_final, 0, 255)

    hsv_small = cv2.merge([Hch, Sch, V_final]).astype(np.uint8)
    bgr_small = cv2.cvtColor(hsv_small, cv2.COLOR_HSV2BGR)

    # expande cada módulo para um bloco module_size x module_size
    out = cv2.resize(bgr_small, (W * module_size, H * module_size), interpolation=cv2.INTER_NEAREST)
    return out

# ======================================================
# (SE QUISER COMPARAR) — SEU MÉTODO ANTIGO EM GRAYSCALE
# ======================================================
# def render_modulated_qr(qr_matrix, perceptual_map, module_size):
#     h, w = qr_matrix.shape
#     img = np.ones((h * module_size, w * module_size), dtype=np.uint8) * 255
#     for y in range(h):
#         for x in range(w):
#             block = img[y*module_size:(y+1)*module_size, x*module_size:(x+1)*module_size]
#             p = perceptual_map[y, x]
#             if qr_matrix[y, x]:  # preto
#                 intensity = int(10 + 170 * p)
#             else:                # branco
#                 intensity = int(255 - 160 * p)
#             block[:] = np.clip(intensity, 0, 255)
#     return img.astype(np.uint8)

# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":

    print("Iniciando - QR com modulacao HSV + alpha adaptativo")

    print("Imagem base encontrada:", os.path.exists(BASE_IMAGE_PATH))
    print("Diretorio de saida:", OUTPUT_DIR)

    qr_matrix = generate_qr_matrix(DATA, QR_VERSION, ERROR_CORRECTION)
    print("Dimensao do QR (modulos):", qr_matrix.shape)

    base_bgr_small, base_gray_small = preprocess_base_image_color(BASE_IMAGE_PATH, qr_matrix.shape)

    edge_map = edge_enhanced_map(base_gray_small)
    perceptual = perceptual_map(base_gray_small, edge_map, gamma=0.70)

    tex = texture_map(base_gray_small, ksize=3)

    # Render final colorido
    final_qr_bgr = render_modulated_qr_hsv(
        qr_matrix=qr_matrix,
        base_bgr_small=base_bgr_small,
        perceptual_small=perceptual,
        texture_small=tex,
        #function_mask=function_mask,   # se a sua função tiver isso, mantenha!
        module_size=MODULE_SIZE,

        alpha_min=0.20,
        alpha_max=0.72,

        v_black=65,
        v_white=255,

        s_boost=1.05,

        # se sua versão tiver clamps:
        #black_v_max=150,
        #white_v_min=235
    )

    ok1 = cv2.imwrite(OUTPUT_QR, final_qr_bgr)
    plt.imsave(OUTPUT_DEBUG, perceptual, cmap="gray")

    print("QR salvo com sucesso?", ok1)
    print("Mapa perceptual salvo:", os.path.exists(OUTPUT_DEBUG))
    print("Arquivo QR:", OUTPUT_QR)
    print("Processamento finalizado com sucesso.")
