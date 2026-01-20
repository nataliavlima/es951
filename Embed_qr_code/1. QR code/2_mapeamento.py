import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# 2.1 Carregar e binarizar a imagem

def load_and_binarize_qr(image_path, threshold=127):
    """
    Carrega a imagem do QR Code e a converte para binário.
    Retorna a imagem binarizada.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img_bin

# 2.2 Remover a quiet zone

def remove_quiet_zone(img_bin, num_modules, border_modules=1):
    """
    Remove a quiet zone do QR Code.
    """
    h, w = img_bin.shape
    module_size = h // (num_modules + 2 * border_modules)

    qr_no_border = img_bin[
        border_modules * module_size : -border_modules * module_size,
        border_modules * module_size : -border_modules * module_size
    ]

    return qr_no_border, module_size

# 2.3 Converter para matriz de módulos

def extract_module_matrix(qr_no_border, num_modules, module_size):
    """
    Converte a imagem do QR Code em uma matriz lógica de módulos.
    1 = preto, 0 = branco
    """
    qr_matrix = np.zeros((num_modules, num_modules), dtype=np.uint8)

    for i in range(num_modules):
        for j in range(num_modules):
            block = qr_no_border[
                i * module_size : (i + 1) * module_size,
                j * module_size : (j + 1) * module_size
            ]
            qr_matrix[i, j] = 1 if np.mean(block) < 127 else 0

    return qr_matrix

# 2.4 Gerar máscara estrutural (regiões não modificáveis)

def generate_protected_mask(num_modules):
    """
    Gera máscara das regiões estruturais do QR Code.
    1 = região protegida
    0 = região modificável
    """
    mask = np.zeros((num_modules, num_modules), dtype=np.uint8)

    finder_size = 7
    sep_size = finder_size + 1

    # Finder patterns + separadores
    mask[0:sep_size, 0:sep_size] = 1
    mask[0:sep_size, -sep_size:] = 1
    mask[-sep_size:, 0:sep_size] = 1

    # Timing patterns
    mask[6, :] = 1
    mask[:, 6] = 1

    return mask

# 2.5 Pipeline completo (função principal)

def map_qr_code(image_path, num_modules, border_modules=1):
    """
    Pipeline completo de mapeamento do QR Code.
    Retorna:
    - matriz lógica do QR
    - máscara estrutural
    - tamanho do módulo
    """
    img_bin = load_and_binarize_qr(image_path)
    qr_no_border, module_size = remove_quiet_zone(
        img_bin, num_modules, border_modules
    )
    qr_matrix = extract_module_matrix(
        qr_no_border, num_modules, module_size
    )
    protected_mask = generate_protected_mask(num_modules)

    return qr_matrix, protected_mask, module_size

# 2.6 Visualização da máscara 

def visualize_mask(qr_matrix, mask):
    """
    Visualiza a máscara estrutural sobre o QR Code.
    """
    vis = qr_matrix.copy()
    vis[mask == 1] = 2
    return vis

## Aplicação nos 9 QR CODES de teste

base_path = "qr_referencia"

qr_configs = {
    "qr_v2_ecL.png": 25,
    "qr_v4_ecL.png": 33,
    "qr_v4_ecM.png": 33,
    "qr_v4_ecQ.png": 33,
    "qr_v4_ecH.png": 33,
    "qr_v6_ecL.png": 41,
    "qr_v6_ecM.png": 41,
    "qr_v6_ecQ.png": 41,
    "qr_v6_ecH.png": 41,
}

mapped_qrs = {}

for filename, num_modules in qr_configs.items():
    image_path = os.path.join(base_path, filename)

    qr_matrix, mask, module_size = map_qr_code(
        image_path=image_path,
        num_modules=num_modules,
        border_modules=1
    )

    mapped_qrs[filename] = {
        "qr_matrix": qr_matrix,
        "mask": mask,
        "module_size": module_size
    }

    print(f"QR Code mapeado com sucesso: {filename}")


# Exemplo de validação visual

example_key = "qr_v6_ecL.png"

example = mapped_qrs[example_key]
visual = visualize_mask(example["qr_matrix"], example["mask"])

plt.figure(figsize=(5, 5))
plt.imshow(visual, cmap="gray")
plt.title(f"Máscara estrutural — {example_key}")
plt.axis("off")
plt.show()