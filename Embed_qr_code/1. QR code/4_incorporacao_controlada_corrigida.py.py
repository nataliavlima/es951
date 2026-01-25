import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURAÇÃO
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QR_PATH = os.path.join(
    BASE_DIR, "qr_referencia", "qr_v10_ecH.png"
)

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR, "imagens_base", "FERB.jpg"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "resultado_qr_etapa4.png"
)

NUM_MODULES = 49  # versão 8 → 49 módulos

# ======================================================
# FUNÇÕES DO QR
# ======================================================

def load_and_binarize_qr(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("QR não encontrado.")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def remove_quiet_zone(img, num_modules, border=1):
    h, _ = img.shape
    module_size = h // (num_modules + 2 * border)
    start = border * module_size
    end = start + num_modules * module_size
    return img[start:end, start:end], module_size


def qr_to_matrix(qr_img, num_modules, module_size):
    matrix = np.zeros((num_modules, num_modules), dtype=np.uint8)
    for y in range(num_modules):
        for x in range(num_modules):
            block = qr_img[
                y*module_size:(y+1)*module_size,
                x*module_size:(x+1)*module_size
            ]
            matrix[y, x] = 1 if np.mean(block) < 127 else 0
    return matrix


def protected_mask(num_modules):
    mask = np.zeros((num_modules, num_modules), dtype=np.uint8)

    def finder(x, y):
        mask[y:y+9, x:x+9] = 1

    finder(0, 0)
    finder(num_modules-9, 0)
    finder(0, num_modules-9)

    mask[6, :] = 1
    mask[:, 6] = 1

    return mask


# ======================================================
# FUNÇÕES DA IMAGEM BASE (ETAPA 3)
# ======================================================

def preprocess_base(image_path, num_modules):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = min(h, w)
    img = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
    img = cv2.resize(img, (num_modules, num_modules))
    return img.astype(np.float32) / 255.0


def contrast_map(luminance):
    lap = cv2.Laplacian(luminance, cv2.CV_32F)
    c = np.abs(lap)
    return c / (c.max() + 1e-8)


# ======================================================
# ETAPA 4 — INCORPORAÇÃO CONTROLADA
# ======================================================

def controlled_embedding(qr, lum, cont, mask,
                          alpha=0.6, beta=0.4, threshold=0.5):

    embedded = qr.copy()

    for y in range(qr.shape[0]):
        for x in range(qr.shape[1]):

            if mask[y, x]:
                continue

            weight = alpha * cont[y, x] + beta * abs(lum[y, x] - 0.5)

            if weight > threshold:
                embedded[y, x] = 1 if lum[y, x] < 0.5 else 0

    return embedded


def render_qr(matrix, module_size):
    size = matrix.shape[0] * module_size
    img = np.ones((size, size), dtype=np.uint8) * 255

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x]:
                img[
                    y*module_size:(y+1)*module_size,
                    x*module_size:(x+1)*module_size
                ] = 0
    return img


# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":

    print("Iniciando Etapa 4 — Incorporação Controlada")

    qr_img = load_and_binarize_qr(QR_PATH)
    qr_crop, module_size = remove_quiet_zone(qr_img, NUM_MODULES)
    qr_matrix = qr_to_matrix(qr_crop, NUM_MODULES, module_size)

    mask = protected_mask(NUM_MODULES)

    lum = preprocess_base(BASE_IMAGE_PATH, NUM_MODULES)
    cont = contrast_map(lum)

    embedded = controlled_embedding(qr_matrix, lum, cont, mask)

    final = render_qr(embedded, module_size)

    cv2.imwrite(OUTPUT_PATH, final)
    plt.imsave("visualizacao_etapa4.png", final, cmap="gray")

    print("Etapa 4 finalizada.")
    print("Arquivo gerado:", OUTPUT_PATH)
