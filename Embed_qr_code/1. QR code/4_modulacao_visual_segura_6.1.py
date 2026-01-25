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

BORDER = 4  # quiet zone (módulos)

# Duas imagens para a análise
IMAGES = [
    "FERB.jpg",
    "homer.png",
]

# Saída (subpastas por imagem)
OUTPUT_ROOT = os.path.join(BASE_DIR, "resultados_6.1")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ======================================================
# GRID DE PARÂMETROS (combinações)
# Mantive o MESMO grid enxuto do código 1 (16 por imagem),
# e deixei os clamps fixos (bvm/wvm) para não explodir o total.
# ======================================================

GRID = {
    "module_size": [12, 16],
    "alpha_min":   [0.10, 0.15],
    "alpha_max":   [0.60, 0.80],
    "v_black":     [40, 80],
    "v_white":     [255],
    "s_boost":     [1.10],
    "gamma":       [0.70],
    # clamps (fixos neste lote; você pode transformar em [120,150] etc. depois)
    "black_v_max": [140],
    "white_v_min": [210],
}

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
    return np.array(qr.get_matrix(), dtype=np.uint8)  # 1 = preto

# ======================================================
# MÁSCARA DE MÓDULOS FUNCIONAIS (NÃO MODULAR)
# ======================================================

def alignment_positions(version):
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
    for i in range(num - 2, 0, -1):
        middle.append(last - step * i)

    return [6] + middle + [last]

def build_function_module_mask(version, border):
    n = 17 + 4 * version
    total = n + 2 * border
    mask = np.zeros((total, total), dtype=bool)

    # quiet zone toda protegida
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True

    off = border

    def mark_rect(x0, y0, w, h):
        mask[y0:y0+h, x0:x0+w] = True

    # finder + separadores (9x9)
    mark_rect(off + 0,     off + 0,     9, 9)
    mark_rect(off + n-8,   off + 0,     9, 9)
    mark_rect(off + 0,     off + n-8,   9, 9)

    # timing
    y_t = off + 6
    x_t = off + 6
    mask[y_t, off+8:off+(n-8)] = True
    mask[off+8:off+(n-8), x_t] = True

    # dark module fixo
    dark_r = 4 * version + 9
    dark_c = 8
    mask[off + dark_r, off + dark_c] = True

    # format info (aproximação conservadora)
    mask[off + 8, off + 0:off + 9] = True
    mask[off + 8, off + (n-8):off + n] = True
    mask[off + 0:off + 9, off + 8] = True
    mask[off + (n-8):off + n, off + 8] = True

    # version info (v>=7)
    if version >= 7:
        mark_rect(off + (n-11), off + 0, 3, 6)
        mark_rect(off + 0, off + (n-11), 6, 3)

    # alignment patterns (5x5)
    pos = alignment_positions(version)
    if pos:
        for cy in pos:
            for cx in pos:
                if (cx <= 8 and cy <= 8) or (cx >= n-9 and cy <= 8) or (cx <= 8 and cy >= n-9):
                    continue
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
        raise ValueError(f"Imagem base nao encontrada: {image_path}")

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
    return np.clip(mag / p95, 0.0, 1.0).astype(np.float32)

# ======================================================
# RENDER HSV + ALPHA ADAPTATIVO + PROTEÇÃO FUNCIONAL
# ======================================================

def render_modulated_qr_hsv(
    qr_matrix,
    base_bgr_small,
    perceptual_small,
    texture_small,
    function_mask,
    module_size,
    alpha_min,
    alpha_max,
    v_black,
    v_white,
    s_boost,
    black_v_max,
    white_v_min
):
    base_hsv = cv2.cvtColor(base_bgr_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    Hch, Sch, Vch = cv2.split(base_hsv)
    Sch = np.clip(Sch * s_boost, 0, 255)

    alpha = alpha_max - (alpha_max - alpha_min) * texture_small
    alpha = np.clip(alpha * (1.0 - 0.18 * perceptual_small), 0.05, 0.95).astype(np.float32)

    # protege módulos funcionais (força alvo do QR nesses locais)
    alpha = np.where(function_mask, 1.0, alpha).astype(np.float32)

    targetV = np.where(qr_matrix == 1, v_black, v_white).astype(np.float32)
    V_final = (1.0 - alpha) * Vch + alpha * targetV

    # clamps de contraste
    V_final = np.where(qr_matrix == 1, np.minimum(V_final, black_v_max), V_final)
    V_final = np.where(qr_matrix == 0, np.maximum(V_final, white_v_min), V_final)

    # quiet zone garantida branca
    V_final = np.where(function_mask & (qr_matrix == 0), 255.0, V_final)
    V_final = np.clip(V_final, 0, 255)

    hsv_small = cv2.merge([Hch, Sch, V_final]).astype(np.uint8)
    bgr_small = cv2.cvtColor(hsv_small, cv2.COLOR_HSV2BGR)

    H, W = qr_matrix.shape
    out = cv2.resize(bgr_small, (W * module_size, H * module_size), interpolation=cv2.INTER_NEAREST)
    return out

# ======================================================
# UTIL — nome do arquivo na ordem pedida
# qr_<stem>_ms12_a0.10-0.80_vb40_vw255_sb1.10_g0.70_bvm140_wvm210.png
# ======================================================

def make_tag(p):
    return (
        f"ms{p['module_size']}"
        f"_a{p['alpha_min']:.2f}-{p['alpha_max']:.2f}"
        f"_vb{p['v_black']}"
        f"_vw{p['v_white']}"
        f"_sb{p['s_boost']:.2f}"
        f"_g{p['gamma']:.2f}"
        f"_bvm{p['black_v_max']}"
        f"_wvm{p['white_v_min']}"
    )

# ======================================================
# EXECUÇÃO EM LOTE (FERB + Homer)
# ======================================================

if __name__ == "__main__":
    print("Iniciando - Batch QR HSV com function_mask + clamps (codigo 2)")

    qr_matrix = generate_qr_matrix(DATA, QR_VERSION, ERROR_CORRECTION, BORDER)
    function_mask = build_function_module_mask(QR_VERSION, BORDER)

    keys = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    print(f"Total de combinacoes: {len(combos)} por imagem (x {len(IMAGES)} imagens) = {len(combos) * len(IMAGES)}")
    print("Dimensao do QR (modulos, incluindo border):", qr_matrix.shape)

    det = cv2.QRCodeDetector()

    for img_name in IMAGES:
        image_path = os.path.join(BASE_DIR, "imagens_base", img_name)
        if not os.path.exists(image_path):
            print(f"[ERRO] Imagem nao encontrada: {image_path}")
            continue

        stem = os.path.splitext(img_name)[0]
        out_dir = os.path.join(OUTPUT_ROOT, stem)
        os.makedirs(out_dir, exist_ok=True)

        base_bgr_small, base_gray_small = preprocess_base_image_color(image_path, qr_matrix.shape)
        edge_map = edge_enhanced_map(base_gray_small)

        print(f"\nProcessando: {img_name} -> {out_dir}")

        for values in combos:
            p = dict(zip(keys, values))

            perceptual = perceptual_map(base_gray_small, edge_map, gamma=p["gamma"])
            tex = texture_map(base_gray_small, ksize=3)

            final_qr_bgr = render_modulated_qr_hsv(
                qr_matrix=qr_matrix,
                base_bgr_small=base_bgr_small,
                perceptual_small=perceptual,
                texture_small=tex,
                function_mask=function_mask,
                module_size=p["module_size"],
                alpha_min=p["alpha_min"],
                alpha_max=p["alpha_max"],
                v_black=p["v_black"],
                v_white=p["v_white"],
                s_boost=p["s_boost"],
                black_v_max=p["black_v_max"],
                white_v_min=p["white_v_min"],
            )

            tag = make_tag(p)
            out_png = os.path.join(out_dir, f"qr_{stem}_{tag}.png")
            out_dbg = os.path.join(out_dir, f"percept_{stem}_{tag}.png")

            ok = cv2.imwrite(out_png, final_qr_bgr)
            plt.imsave(out_dbg, perceptual, cmap="gray")

            # (Opcional) auto-teste de decode para log rápido no console
            decoded, _, _ = det.detectAndDecode(final_qr_bgr)
            if not ok:
                print(f"[ERRO] Falhou ao salvar: {out_png}")
            # descomente se quiser logar tudo:
            # print("OK" if decoded else "FAIL", os.path.basename(out_png))

        print(f"Finalizado: {img_name}")

    print("\nBatch finalizado.")
