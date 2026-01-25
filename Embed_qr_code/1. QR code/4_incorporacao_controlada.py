import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURAÇÃO DE CAMINHOS (ROBUSTO)
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QR_PATH = os.path.join(
    BASE_DIR,
    "qr_referencia",
    "qr_v10_ecH.png"
)

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "imagens_base",
    #"bob.png"
    #"FERB.jpg"
    "homer.png"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "resultado_qr_estilizado_3.png"
)

# ======================================================
# PARTE 2 — MAPEAMENTO DO QR CODE
# ======================================================

def load_and_binarize_qr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("QR Code de referência não encontrado.")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def remove_quiet_zone(img, num_modules, border_modules):
    h, w = img.shape
    module_size = h // (num_modules + 2 * border_modules)
    start = border_modules * module_size
    end = start + num_modules * module_size
    qr_no_border = img[start:end, start:end]
    return qr_no_border, module_size


def extract_module_matrix(qr_img, num_modules, module_size):
    matrix = np.zeros((num_modules, num_modules), dtype=np.uint8)
    for y in range(num_modules):
        for x in range(num_modules):
            block = qr_img[
                y * module_size:(y + 1) * module_size,
                x * module_size:(x + 1) * module_size
            ]
            matrix[y, x] = 1 if np.mean(block) < 127 else 0
    return matrix


def generate_protected_mask(num_modules):
    mask = np.zeros((num_modules, num_modules), dtype=np.uint8)

    def protect_finder(x, y):
        mask[y:y+9, x:x+9] = 1

    protect_finder(0, 0)
    protect_finder(num_modules - 9, 0)
    protect_finder(0, num_modules - 9)

    mask[6, :] = 1
    mask[:, 6] = 1

    return mask


def map_qr_code(image_path, num_modules, border_modules=1):
    img_bin = load_and_binarize_qr(image_path)
    qr_no_border, module_size = remove_quiet_zone(
        img_bin, num_modules, border_modules
    )
    qr_matrix = extract_module_matrix(
        qr_no_border, num_modules, module_size
    )
    protected_mask = generate_protected_mask(num_modules)
    return qr_matrix, protected_mask, module_size


# ======================================================
# PARTE 3.1 — PRÉ-PROCESSAMENTO DA IMAGEM BASE
# ======================================================

def load_image_gray(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Imagem base não encontrada.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def center_crop_square(image):
    h, w = image.shape
    side = min(h, w)
    start_y = (h - side) // 2
    start_x = (w - side) // 2
    return image[start_y:start_y + side, start_x:start_x + side]


def preprocess_base_image(image_path, num_modules):
    gray = load_image_gray(image_path)
    square = center_crop_square(gray)
    resized = cv2.resize(
        square,
        (num_modules, num_modules),
        interpolation=cv2.INTER_AREA
    )
    normalized = resized.astype(np.float32) / 255.0
    return normalized


# ======================================================
# PARTE 3.2 — INCORPORAÇÃO (HALFTONE BINÁRIO SIMPLES)
# ======================================================

def render_qr_image(matrix, module_size):
    size = matrix.shape[0] * module_size
    img = np.ones((size, size), dtype=np.uint8) * 255

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] == 1:
                img[
                    y * module_size:(y + 1) * module_size,
                    x * module_size:(x + 1) * module_size
                ] = 0
    return img

def embed_image(qr_matrix, base_img, protected_mask,
                           img_threshold=0.5, delta=0.15):
    """
    Incorporação controlada da imagem base no QR Code.

    img_threshold : limiar de luminância da imagem
    delta         : margem de confiança para alterar o módulo
    """

    embedded = qr_matrix.copy()

    for y in range(qr_matrix.shape[0]):
        for x in range(qr_matrix.shape[1]):

            if protected_mask[y, x] == 1:
                continue

            img_val = base_img[y, x]
            qr_val = qr_matrix[y, x]

            # Região claramente escura na imagem
            if img_val < (img_threshold - delta):
                embedded[y, x] = 1  # força preto

            # Região claramente clara na imagem
            elif img_val > (img_threshold + delta):
                embedded[y, x] = 0  # força branco

            # Região ambígua → mantém QR original
            else:
                embedded[y, x] = qr_val

    return embedded

def embed_image_fusion(qr_matrix, base_img, protected_mask,
                       alpha=0.7):
    """
    Incorporação por fusão contínua simples (Garateguy et al.)

    alpha : peso do QR Code (0.6–0.8 recomendado)
    beta  : peso da imagem base (1 - alpha)
    """

    beta = 1.0 - alpha
    embedded = qr_matrix.copy().astype(np.float32)

    for y in range(qr_matrix.shape[0]):
        for x in range(qr_matrix.shape[1]):

            if protected_mask[y, x] == 1:
                continue

            qr_val = qr_matrix[y, x]
            img_val = base_img[y, x]

            combined = alpha * qr_val + beta * (1.0 - img_val)

            embedded[y, x] = 1 if combined > 0.5 else 0

    return embedded.astype(np.uint8)

# ======================================================
# EXECUÇÃO DO EXPERIMENTO
# ======================================================

if __name__ == "__main__":

    print("Iniciando processamento do QR Code estilizado...")
    print("QR encontrado:", os.path.exists(QR_PATH))
    print("Imagem base encontrada:", os.path.exists(BASE_IMAGE_PATH))

    num_modules = 33  # Versão 4 do QR Code

    qr_matrix, protected_mask, module_size = map_qr_code(
        QR_PATH, num_modules
    )

    base_img = preprocess_base_image(
        BASE_IMAGE_PATH, num_modules
    )

    embedded_matrix = embed_image_fusion(
    qr_matrix, base_img, protected_mask,
    alpha=0.6
    )


    final_qr = render_qr_image(
        embedded_matrix, module_size
    )

    cv2.imwrite(OUTPUT_PATH, final_qr)
    plt.imsave("visualizacao_qr.png", final_qr, cmap="gray")

    print("Processamento finalizado com sucesso.")
    print("Arquivo gerado em:", OUTPUT_PATH)
