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
QR_VERSION = 8  # versão grande → mais espaço visual
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR, "imagens_base", "FERB.jpg"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# ETAPA 1 — GERAR MATRIZ DO QR (SEM PIXEL, SEM RECORTE)
# ======================================================

def generate_qr_matrix(data, version, error_correction):
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=1,
        border=4  # quiet zone padrão
    )
    qr.add_data(data)
    qr.make(fit=False)

    matrix = np.array(qr.get_matrix(), dtype=np.uint8)
    return matrix


# ======================================================
# ETAPA 2 — RENDERIZAR QR PARA VISUALIZAÇÃO
# ======================================================

def render_qr_from_matrix(matrix, module_size=10):
    h, w = matrix.shape
    img = np.ones((h * module_size, w * module_size), dtype=np.uint8) * 255

    for y in range(h):
        for x in range(w):
            if matrix[y, x]:
                img[
                    y * module_size:(y + 1) * module_size,
                    x * module_size:(x + 1) * module_size
                ] = 0
    return img


# ======================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO DA IMAGEM BASE (MAPA CONTÍNUO)
# ======================================================

def preprocess_base_image(image_path, target_shape):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem base não encontrada.")

    h, w = img.shape
    side = min(h, w)

    img = img[
        (h - side) // 2:(h + side) // 2,
        (w - side) // 2:(w + side) // 2
    ]

    img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    return img


# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":

    print("=== BASELINE DO EXPERIMENTO — ETAPAS 1 A 3 ===")

    # ---- QR lógico
    qr_matrix = generate_qr_matrix(
        DATA, QR_VERSION, ERROR_CORRECTION
    )

    print("Dimensão do QR (módulos):", qr_matrix.shape)

    qr_img = render_qr_from_matrix(qr_matrix, module_size=10)
    qr_path = os.path.join(OUTPUT_DIR, "qr_baseline.png")
    cv2.imwrite(qr_path, qr_img)

    print("QR baseline salvo em:", qr_path)
    print("Teste este QR no celular. Ele DEVE funcionar.")

    # ---- Imagem base como mapa contínuo
    base_map = preprocess_base_image(
        BASE_IMAGE_PATH, qr_matrix.shape
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(base_map, cmap="gray")
    plt.title("Mapa contínuo da imagem base (Etapa 3)")
    plt.axis("off")

    base_vis_path = os.path.join(
        OUTPUT_DIR, "mapa_continuo_base.png"
    )
    plt.savefig(base_vis_path, bbox_inches="tight")
    plt.close()

    print("Mapa contínuo salvo em:", base_vis_path)
    print(" Aqui você deve reconhecer claramente o Ferb.")

    print("=== BASELINE FINALIZADO COM SUCESSO ===")
