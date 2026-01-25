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
QR_VERSION = 10                     # versão grande
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H
MODULE_SIZE = 10                   # tamanho visual do módulo

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR, "imagens_base", "FERB.jpg" #"homer.png"  # ou FERB.jpg
)

OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_QR = os.path.join(
    OUTPUT_DIR, "qr_modulado_perceptual.png"
)

OUTPUT_DEBUG = os.path.join(
    OUTPUT_DIR, "mapa_perceptual.png"
)

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
    return np.array(qr.get_matrix(), dtype=np.uint8)

# ======================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO DA IMAGEM BASE
# ======================================================

def preprocess_base_image(image_path, target_shape):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem base nao encontrada.")

    h, w = img.shape
    side = min(h, w)

    img = img[
        (h - side) // 2:(h + side) // 2,
        (w - side) // 2:(w + side) // 2
    ]

    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)

    # normalização
    img = img.astype(np.float32) / 255.0

    # leve aumento de contraste
    img = cv2.equalizeHist((img * 255).astype(np.uint8)).astype(np.float32) / 255.0

    return img

# ======================================================
# MELHORIA PERCEPTUAL — CONTORNOS
# ======================================================

def edge_enhanced_map(base_map):
    edges = cv2.Canny(
        (base_map * 255).astype(np.uint8),
        threshold1=60,
        threshold2=140
    )
    edges = edges.astype(np.float32) / 255.0
    return edges

def perceptual_map(base_map, edge_map, gamma=0.6):
    combined = gamma * base_map + (1.0 - gamma) * edge_map
    return np.clip(combined, 0.0, 1.0)

# ======================================================
# ETAPA 4 — MODULAÇÃO VISUAL SEGURA
# (não altera a lógica do QR)
# ======================================================

def render_modulated_qr(qr_matrix, perceptual_map, module_size):
    h, w = qr_matrix.shape
    img = np.ones((h * module_size, w * module_size), dtype=np.uint8) * 255

    for y in range(h):
        for x in range(w):

            block = img[
                y * module_size:(y + 1) * module_size,
                x * module_size:(x + 1) * module_size
            ]

            p = perceptual_map[y, x]

            if qr_matrix[y, x]:  # módulo preto
                intensity = int(10 + 170 * p)
            else:                # módulo branco
                intensity = int(255 - 160 * p)

            block[:] = np.clip(intensity, 0, 255)

    return img.astype(np.uint8)

# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":

    print("Iniciando Etapa 4 - Modulacao perceptual segura")

    print("Imagem base encontrada:", os.path.exists(BASE_IMAGE_PATH))
    print("Diretorio de saida:", OUTPUT_DIR)

    # QR lógico
    qr_matrix = generate_qr_matrix(
        DATA, QR_VERSION, ERROR_CORRECTION
    )

    print("Dimensao do QR (modulos):", qr_matrix.shape)

    # Imagem base
    base_map = preprocess_base_image(
        BASE_IMAGE_PATH, qr_matrix.shape
    )

    # Mapas perceptuais
    edge_map = edge_enhanced_map(base_map)
    perceptual = perceptual_map(base_map, edge_map)

    # QR modulado
    final_qr = render_modulated_qr(
        qr_matrix, perceptual, MODULE_SIZE
    )

    # DEBUG DA IMAGEM FINAL
    print("Imagem final:")
    print(" - shape:", final_qr.shape)
    print(" - dtype:", final_qr.dtype)
    print(" - min/max:", final_qr.min(), final_qr.max())

    # Salvando resultados (com verificação!)
    ok1 = cv2.imwrite(OUTPUT_QR, final_qr)
    ok2 = plt.imsave(OUTPUT_DEBUG, perceptual, cmap="gray")

    print("QR salvo com sucesso?", ok1)
    print("Mapa perceptual salvo:", os.path.exists(OUTPUT_DEBUG))
    print("Arquivo QR:", OUTPUT_QR)

    print("Processamento finalizado com sucesso.")
