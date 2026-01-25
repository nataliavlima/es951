import os
import cv2
import numpy as np
import qrcode
import qrcode.constants
import matplotlib.pyplot as plt
from itertools import product

# ======================================================
# CONFIGURAÇÃO
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA = "https://www.fem.unicamp.br/"
QR_VERSION = 10
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H

BORDER = 4  # quiet zone (módulos) - usado no qrcode
IMAGES = [
    "FERB.jpg",
    "homer.png",
]

# Diretório raiz de saída (vai criar subpastas por imagem)
OUTPUT_ROOT = os.path.join(BASE_DIR, "resultados_5.1")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ======================================================
# GRID DE PARÂMETROS (combinações)
# (mantendo o grid enxuto: 8 configurações por imagem)
# ======================================================

GRID = {
    "module_size": [12, 16],     # tamanho visual do módulo
    "alpha_min":   [0.10, 0.15], # força mínima
    "alpha_max":   [0.60, 0.80], # força máxima
    "v_black":     [40, 80],     # "preto" mais claro/escuro
    "v_white":     [255],        # fixo em branco alto para leitura
    "s_boost":     [1.05],       # fixo (pode adicionar 1.10 depois)
    "gamma":       [0.70],       # fixo (pode variar 0.6/0.8 depois)
}

# ======================================================
# ETAPA 1 — GERAR MATRIZ LÓGICA DO QR
# ======================================================

def generate_qr_matrix(data, version, error_correction, border=BORDER):
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
        raise ValueError(f"Imagem base nao encontrada: {image_path}")

    img_bgr = center_crop_square(img_bgr)
    H, W = target_shape
    img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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

def perceptual_map(base_gray_float, edge_map, gamma=0.70):
    combined = gamma * base_gray_float + (1.0 - gamma) * edge_map
    return np.clip(combined, 0.0, 1.0)

# ======================================================
# MAPA DE TEXTURA (para alpha adaptativo)
# ======================================================

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
# ETAPA 4 — RENDERIZAÇÃO HSV (S e V) + ALPHA ADAPTATIVO
# ======================================================

def render_modulated_qr_hsv(
    qr_matrix,
    base_bgr_small,     # (H,W,3) 0..255
    perceptual_small,   # (H,W) 0..1
    texture_small,      # (H,W) 0..1
    module_size,
    alpha_min=0.20,
    alpha_max=0.72,
    v_black=65,
    v_white=255,
    s_boost=1.05
):
    H, W = qr_matrix.shape

    base_hsv = cv2.cvtColor(base_bgr_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    Hch, Sch, Vch = cv2.split(base_hsv)

    Sch = np.clip(Sch * s_boost, 0, 255)

    alpha = alpha_max - (alpha_max - alpha_min) * texture_small
    alpha = np.clip(alpha * (1.0 - 0.18 * perceptual_small), 0.05, 0.95).astype(np.float32)

    targetV = np.where(qr_matrix == 1, v_black, v_white).astype(np.float32)
    V_final = (1.0 - alpha) * Vch + alpha * targetV
    V_final = np.clip(V_final, 0, 255)

    hsv_small = cv2.merge([Hch, Sch, V_final]).astype(np.uint8)
    bgr_small = cv2.cvtColor(hsv_small, cv2.COLOR_HSV2BGR)

    out = cv2.resize(bgr_small, (W * module_size, H * module_size), interpolation=cv2.INTER_NEAREST)
    return out

# ======================================================
# UTIL — nome de arquivo padronizado
# ======================================================

def make_tag(params):
    # tag curta, mas informativa
    return (
        f"ms{params['module_size']}"
        f"_a{params['alpha_min']:.2f}-{params['alpha_max']:.2f}"
        f"_vb{params['v_black']}"
        f"_vw{params['v_white']}"
        f"_sb{params['s_boost']:.2f}"
        f"_g{params['gamma']:.2f}"
    )

# ======================================================
# EXECUÇÃO EM LOTE
# ======================================================

if __name__ == "__main__":
    print("Iniciando - Batch QR com modulacao HSV + alpha adaptativo")

    qr_matrix = generate_qr_matrix(DATA, QR_VERSION, ERROR_CORRECTION, border=BORDER)
    print("Dimensao do QR (modulos):", qr_matrix.shape)

    # prepara combinações do grid
    keys = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    print(f"Total de combinacoes: {len(combos)} por imagem (x {len(IMAGES)} imagens) = {len(combos)*len(IMAGES)}")

    for img_name in IMAGES:
        image_path = os.path.join(BASE_DIR, "imagens_base", img_name)
        if not os.path.exists(image_path):
            print(f"[ERRO] Imagem nao encontrada: {image_path}")
            continue

        # subpasta por imagem
        stem = os.path.splitext(img_name)[0]
        out_dir = os.path.join(OUTPUT_ROOT, stem)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessando imagem: {img_name}")

        base_bgr_small, base_gray_small = preprocess_base_image_color(image_path, qr_matrix.shape)
        edge_map = edge_enhanced_map(base_gray_small)

        # para cada combinação
        for values in combos:
            params = dict(zip(keys, values))

            perceptual = perceptual_map(base_gray_small, edge_map, gamma=params["gamma"])
            tex = texture_map(base_gray_small, ksize=3)

            final_qr_bgr = render_modulated_qr_hsv(
                qr_matrix=qr_matrix,
                base_bgr_small=base_bgr_small,
                perceptual_small=perceptual,
                texture_small=tex,
                module_size=params["module_size"],
                alpha_min=params["alpha_min"],
                alpha_max=params["alpha_max"],
                v_black=params["v_black"],
                v_white=params["v_white"],
                s_boost=params["s_boost"]
            )

            tag = make_tag(params)
            out_png = os.path.join(out_dir, f"qr_{stem}_{tag}.png")
            out_dbg = os.path.join(out_dir, f"percept_{stem}_{tag}.png")

            ok = cv2.imwrite(out_png, final_qr_bgr)
            # mapa perceptual para depuração (opcional, mas útil)
            plt.imsave(out_dbg, perceptual, cmap="gray")

            if not ok:
                print(f"[ERRO] Falhou ao salvar: {out_png}")

        print(f"Finalizado: {img_name} -> {out_dir}")

    print("\nBatch finalizado com sucesso.")
