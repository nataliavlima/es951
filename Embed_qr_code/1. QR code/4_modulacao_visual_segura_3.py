import os
import cv2
import numpy as np
import qrcode
import qrcode.constants

# ======================================================
# CONFIGURAÇÃO
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA = "https://www.fem.unicamp.br/"
QR_VERSION = 8
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR, "imagens_base", "FERB.jpg"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "resultados", "qr_modulado.png"
)

MODULE_SIZE = 10

# ======================================================
# ETAPA 1 — QR LÓGICO
# ======================================================

def generate_qr_matrix(data, version, ec):
    qr = qrcode.QRCode(
        version=version,
        error_correction=ec,
        box_size=1,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=np.uint8)


# ======================================================
# ETAPA 3 — MAPA CONTÍNUO
# ======================================================

def preprocess_base_image(path, target_shape):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = min(h, w)
    img = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
    img = cv2.resize(img, target_shape)
    return img.astype(np.float32) / 255.0


# ======================================================
# ETAPA 4 — MODULAÇÃO VISUAL SEGURA
# ======================================================

def render_modulated_qr(qr_matrix, base_map, module_size):
    h, w = qr_matrix.shape
    img = np.ones((h*module_size, w*module_size), dtype=np.uint8) * 255

    for y in range(h):
        for x in range(w):
            block = img[
                y*module_size:(y+1)*module_size,
                x*module_size:(x+1)*module_size
            ]

            if qr_matrix[y, x]:  # módulo preto
                intensity = int(40 + 100 * base_map[y, x])
            else:  # módulo branco
                intensity = int(255 - 80 * base_map[y, x])

            block[:] = np.clip(intensity, 0, 255)

    return img


# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":

    qr_matrix = generate_qr_matrix(
        DATA, QR_VERSION, ERROR_CORRECTION
    )

    base_map = preprocess_base_image(
        BASE_IMAGE_PATH, qr_matrix.shape
    )

    final_qr = render_modulated_qr(
        qr_matrix, base_map, MODULE_SIZE
    )

    cv2.imwrite(OUTPUT_PATH, final_qr)

    print("QR modulado gerado com sucesso.")
    print("Arquivo:", OUTPUT_PATH)
