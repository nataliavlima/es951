import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# CONFIGURAÇÃO DE CAMINHOS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "imagens_base",
    "FERB.jpg"
)

OUTPUT_LUMINANCE = os.path.join(
    BASE_DIR,
    "mapa_luminancia.png"
)

OUTPUT_CONTRAST = os.path.join(
    BASE_DIR,
    "mapa_contraste.png"
)

# Número de módulos do QR que será usado depois
NUM_MODULES = 33  # ajuste conforme versão do QR

# ======================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO DA IMAGEM BASE
# (SEM ALTERAR O QR CODE)
# ======================================================

def load_image_gray(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Imagem base não encontrada.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def center_crop_square(image):
    h, w = image.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def generate_luminance_map(image_path, num_modules):
    """
    Gera um mapa contínuo de luminância alinhado à grade do QR.
    Valores em [0, 1].
    """
    gray = load_image_gray(image_path)
    square = center_crop_square(gray)

    resized = cv2.resize(
        square,
        (num_modules, num_modules),
        interpolation=cv2.INTER_AREA
    )

    luminance = resized.astype(np.float32) / 255.0
    return luminance


def generate_contrast_map(luminance_map):
    """
    Destaca regiões com variação local (bordas / detalhes).
    """
    laplacian = cv2.Laplacian(luminance_map, cv2.CV_32F)
    contrast = np.abs(laplacian)

    # Normalização para [0, 1]
    contrast = contrast / (contrast.max() + 1e-8)
    return contrast


def save_visualization(map_data, path, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(map_data, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ======================================================
# EXECUÇÃO
# ======================================================

if __name__ == "__main__":

    print("Iniciando ETAPA 3 — Pré-processamento correto")
    print("Imagem base encontrada:", os.path.exists(BASE_IMAGE_PATH))

    luminance_map = generate_luminance_map(
        BASE_IMAGE_PATH,
        NUM_MODULES
    )

    contrast_map = generate_contrast_map(luminance_map)

    save_visualization(
        luminance_map,
        OUTPUT_LUMINANCE,
        "Mapa de Luminância (Imagem Base)"
    )

    save_visualization(
        contrast_map,
        OUTPUT_CONTRAST,
        "Mapa de Contraste (Detalhes Visuais)"
    )

    print("Etapa 3 concluída com sucesso.")
    print("Arquivos gerados:")
    print(" -", OUTPUT_LUMINANCE)
    print(" -", OUTPUT_CONTRAST)
