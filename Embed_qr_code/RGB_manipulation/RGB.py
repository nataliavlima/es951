import cv2
import numpy as np
import qrcode
import os

# === 0. Caminho base da pasta do projeto ===
base_path = os.path.dirname(os.path.abspath(__file__))  # pega a pasta do .py

# === 1. Gerar o QR Code ===
data = "https://www.linkedin.com/in/natalia-vieira-lima-4026bb1a9/"
qr = qrcode.QRCode(
    version=2,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=1,
)
qr.add_data(data)
qr.make(fit=True)

qr_img = qr.make_image(fill_color="black", back_color="white").convert("L")
qr_array = np.array(qr_img)

# Salva o QR
cv2.imwrite(os.path.join(base_path, "qr_gray.png"), qr_array)

# Redimensionar o QR
qr_array = cv2.resize(qr_array, (200, 200), interpolation=cv2.INTER_AREA)

# === 2. Carregar imagem base ===
path = os.path.join(base_path, "bob.jpg")  # imagem na mesma pasta
if not os.path.exists(path):
    raise FileNotFoundError(f"⚠️ Imagem base não encontrada: {path}")

img = cv2.imread(path)
cv2.imwrite(os.path.join(base_path, "original.png"), img)
h, w, _ = img.shape

# Centralizar o QR
x_offset = (w - qr_array.shape[1]) // 2
y_offset = (h - qr_array.shape[0]) // 2

# === 3. Separar canais RGB ===
b, g, r = cv2.split(img)

# === 4. Criar máscara ===
qr_norm = qr_array / 255.0
mask = 1 - qr_norm
alpha = 1

# === 5. Função para aplicar QR em um canal ===
def aplicar_qr(canal):
    canal_mod = canal.copy().astype(np.float32)
    roi = canal_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]]
    roi_new = roi * (1.0 - alpha * mask)
    canal_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]] = np.clip(roi_new, 0, 255)
    return canal_mod.astype(np.uint8)

# === 6. Aplicar em cada canal e salvar ===
r_mod = aplicar_qr(r)
img_R = cv2.merge((b, g, r_mod))
cv2.imwrite(os.path.join(base_path, "saida_qr_R.png"), img_R)

g_mod = aplicar_qr(g)
img_G = cv2.merge((b, g_mod, r))
cv2.imwrite(os.path.join(base_path, "saida_qr_G.png"), img_G)

b_mod = aplicar_qr(b)
img_B = cv2.merge((b_mod, g, r))
cv2.imwrite(os.path.join(base_path, "saida_qr_B.png"), img_B)

# === 7. Mostrar as imagens ===
cv2.imshow("Original", img)
cv2.imshow("QR Gray", qr_array)
cv2.imshow("Modulado R", img_R)
cv2.imshow("Modulado G", img_G)
cv2.imshow("Modulado B", img_B)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Todas as 5 imagens foram salvas em:\n{base_path}")
