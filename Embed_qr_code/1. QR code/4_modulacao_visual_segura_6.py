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

BORDER = 4                 # quiet zone (em módulos)
MODULE_SIZE = 12           # aumente p/ 16 se precisar

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR, "imagens_base",
    #"bob.png"
    #"FERB.jpg" 
    #"homer.png" 
    #"dogs.jpeg" 
    "donald.jpg"
    #"coragem.jpg"
    #"picapau.jpg"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_QR = os.path.join(OUTPUT_DIR, "qr_modulado_perceptual_DONALD.png")
OUTPUT_DEBUG = os.path.join(OUTPUT_DIR, "mapa_perceptual.png")

# ======================================================
# ETAPA 1 — GERAR MATRIZ LÓGICA DO QR (inclui BORDER)
# ======================================================

def generate_qr_matrix(data, version, error_correction, border):
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=1,
        border=border
    )
    qr.add_data(data)
    qr.make(fit=False)
    # qrcode retorna True para módulo PRETO
    return np.array(qr.get_matrix(), dtype=np.uint8)

# ======================================================
# MÁSCARA DE MÓDULOS FUNCIONAIS (NÃO MODULAR)
# ======================================================

def alignment_positions(version):
    """Lista de posições (em coordenadas do QR sem border) para padrões de alinhamento."""
    if version == 1:
        return []
    n = 17 + 4 * version
    num = version // 7 + 2
    if num == 2:
        return [6, n - 7]

    step = (n - 13) // (num - 1)
    if step % 2 == 1:
        step += 1

    last = n - 7
    middle = []
    # gera posições intermediárias de forma estável
    for i in range(num - 2, 0, -1):
        middle.append(last - step * i)

    return [6] + middle + [last]

def build_function_module_mask(version, border):
    """
    Retorna máscara bool (H,W) True onde NÃO devemos modular (módulos funcionais + quiet zone).
    Como o qr_matrix do qrcode inclui border, o tamanho total é:
      total = (17 + 4*version) + 2*border
    """
    n = 17 + 4 * version              # tamanho do QR "puro"
    total = n + 2 * border
    mask = np.zeros((total, total), dtype=bool)

    # Quiet zone inteira: não modular (deve ficar branca)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True

    off = border  # offset para mapear coords do QR puro -> coords totais

    def mark_rect(x0, y0, w, h):
        mask[y0:y0+h, x0:x0+w] = True

    # Finder patterns + separadores (8x8 ao redor de cada 7x7)
    # Top-left
    mark_rect(off + 0,       off + 0,       9, 9)
    # Top-right
    mark_rect(off + (n-8),   off + 0,       9, 9)
    # Bottom-left
    mark_rect(off + 0,       off + (n-8),   9, 9)

    # Timing patterns (linha 6 e coluna 6 no QR puro)
    # elas vão de 8 até n-9 (inclusive) no QR puro; marcamos como funcionais
    y_t = off + 6
    x_t = off + 6
    mask[y_t, off+8:off+(n-8)] = True
    mask[off+8:off+(n-8), x_t] = True

    # "Dark module" fixo: (row = 4*version + 9, col = 8) no QR puro (0-index)
    dark_r = 4 * version + 9
    dark_c = 8
    mask[off + dark_r, off + dark_c] = True

    # Format info (áreas de formato ao redor do top-left + correspondentes)
    # Aproximação conservadora (marcar as regiões usuais)
    # linha 8: col 0..8 e col (n-8)..(n-1)
    mask[off + 8, off + 0:off + 9] = True
    mask[off + 8, off + (n-8):off + n] = True
    # coluna 8: row 0..8 e row (n-8)..(n-1)
    mask[off + 0:off + 9, off + 8] = True
    mask[off + (n-8):off + n, off + 8] = True

    # Version info (somente version >= 7): dois blocos 3x6
    if version >= 7:
        # Top-right: rows 0..5, cols (n-11)..(n-9)  (3 colunas x 6 linhas)
        mark_rect(off + (n-11), off + 0, 3, 6)
        # Bottom-left: rows (n-11)..(n-9), cols 0..5 (6 colunas x 3 linhas)
        mark_rect(off + 0, off + (n-11), 6, 3)

    # Alignment patterns (5x5) para versões >=2
    pos = alignment_positions(version)
    if pos:
        for cy in pos:
            for cx in pos:
                # não marcar os que sobrepõem os finders
                if (cx <= 8 and cy <= 8) or (cx >= n-9 and cy <= 8) or (cx <= 8 and cy >= n-9):
                    continue
                # marca 5x5 em torno do centro
                x0 = off + cx - 2
                y0 = off + cy - 2
                mark_rect(x0, y0, 5, 5)

    return mask

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
# ETAPA 3 — PRÉ-PROCESSAMENTO DA IMAGEM BASE
# ======================================================

def preprocess_base_image_color(image_path, target_shape):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Imagem base nao encontrada.")

    img_bgr = center_crop_square(img_bgr)

    H, W = target_shape
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    base_gray_float = gray_eq.astype(np.float32) / 255.0
    return img_bgr, base_gray_float

# ======================================================
# MAPAS PERCEPTUAIS
# ======================================================

def edge_enhanced_map(base_gray_float):
    edges = cv2.Canny((base_gray_float * 255).astype(np.uint8), 60, 140)
    return edges.astype(np.float32) / 255.0

def perceptual_map(base_gray_float, edge_map, gamma=0.70):
    combined = gamma * base_gray_float + (1.0 - gamma) * edge_map
    return np.clip(combined, 0.0, 1.0)

def texture_map(base_gray_float, ksize=3):
    img = (base_gray_float * 255).astype(np.uint8)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    p95 = np.percentile(mag, 95)
    if p95 <= 1e-6:
        return np.zeros_like(base_gray_float, dtype=np.float32)
    t = np.clip(mag / p95, 0.0, 1.0).astype(np.float32)
    return t

# ======================================================
# RENDER HSV + ALPHA ADAPTATIVO + PROTEÇÃO FUNCIONAL
# ======================================================

def render_modulated_qr_hsv(
    qr_matrix,
    base_bgr_small,
    perceptual_small,
    texture_small,
    function_mask,       # True onde NÃO modular
    module_size,
    alpha_min=0.12,
    alpha_max=0.52,
    v_black=62,
    v_white=255,
    s_boost=1.10,
    # clamps para garantir decodificação
    black_v_max=140,       # preto não pode ficar claro demais
    white_v_min=210       # branco não pode ficar escuro demais
):
    H, W = qr_matrix.shape

    base_hsv = cv2.cvtColor(base_bgr_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    Hch, Sch, Vch = cv2.split(base_hsv)
    Sch = np.clip(Sch * s_boost, 0, 255)

    # alpha adaptativo: liso -> alpha maior (QR mais forte), textura -> alpha menor
    alpha = alpha_max - (alpha_max - alpha_min) * texture_small
    alpha = alpha.astype(np.float32)

    # reforço guiado por perceptual
    alpha = np.clip(alpha * (1.0 - 0.18 * perceptual_small), 0.05, 0.95)

    # módulos funcionais: força alpha=1 (ou seja, V_final = V_target)
    alpha = np.where(function_mask, 1.0, alpha).astype(np.float32)

    targetV = np.where(qr_matrix == 1, v_black, v_white).astype(np.float32)
    V_final = (1.0 - alpha) * Vch + alpha * targetV

    # clamps de segurança (ajuda MUITO em imagens coloridas)
    V_final = np.where(qr_matrix == 1, np.minimum(V_final, black_v_max), V_final)
    V_final = np.where(qr_matrix == 0, np.maximum(V_final, white_v_min), V_final)

    # quiet zone: garantir branco total (borda)
    V_final = np.where(function_mask & (qr_matrix == 0), 255.0, V_final)

    V_final = np.clip(V_final, 0, 255)

    hsv_small = cv2.merge([Hch, Sch, V_final]).astype(np.uint8)
    bgr_small = cv2.cvtColor(hsv_small, cv2.COLOR_HSV2BGR)

    out = cv2.resize(
        bgr_small,
        (W * module_size, H * module_size),
        interpolation=cv2.INTER_NEAREST
    )
    return out

# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":
    print("Iniciando - QR modulacao HSV (com proteção de módulos funcionais)")

    print("Imagem base encontrada:", os.path.exists(BASE_IMAGE_PATH))
    print("Diretorio de saida:", OUTPUT_DIR)

    qr_matrix = generate_qr_matrix(DATA, QR_VERSION, ERROR_CORRECTION, BORDER)
    print("Dimensao do QR (modulos, incluindo border):", qr_matrix.shape)

    # máscara de módulos funcionais
    function_mask = build_function_module_mask(QR_VERSION, BORDER)

    base_bgr_small, base_gray_small = preprocess_base_image_color(BASE_IMAGE_PATH, qr_matrix.shape)
    edge_map = edge_enhanced_map(base_gray_small)
    perceptual = perceptual_map(base_gray_small, edge_map, gamma=0.70)
    tex = texture_map(base_gray_small, ksize=3)

    final_qr_bgr = render_modulated_qr_hsv(
        qr_matrix=qr_matrix,
        base_bgr_small=base_bgr_small,
        perceptual_small=perceptual,
        texture_small=tex,
        function_mask=function_mask,
        module_size=MODULE_SIZE,
        alpha_min=0.12,
        alpha_max=0.52,
        v_black=62,
        v_white=255,
        s_boost=1.10,
        # clamps para garantir decodificação
        black_v_max=140,       # preto não pode ficar claro demais
        white_v_min=210       # branco não pode ficar escuro demais
        )

    ok1 = cv2.imwrite(OUTPUT_QR, final_qr_bgr)
    plt.imsave(OUTPUT_DEBUG, perceptual, cmap="gray")

    print("QR salvo com sucesso?", ok1)
    print("Arquivo QR:", OUTPUT_QR)
    print("Mapa perceptual salvo:", os.path.exists(OUTPUT_DEBUG))

    # Auto-teste de decodificação (muito útil para iterar)
    det = cv2.QRCodeDetector()
    decoded, pts, _ = det.detectAndDecode(final_qr_bgr)
    if decoded:
        print("DECODE OK ->", decoded)
    else:
        print("DECODE FALHOU no teste interno. Tente:")
        print("  - aumentar MODULE_SIZE (12 -> 16)")
        print("  - aumentar alpha_min/alpha_max (ex.: 0.55/0.92)")
        print("  - reduzir black_v_max (ex.: 60) e aumentar white_v_min (ex.: 200)")

    print("Processamento finalizado.")
