
import cv2
import numpy as np
import qrcode
import os

# === 0. Caminho base da pasta do projeto ===
base_path = os.path.dirname(os.path.abspath(__file__))

# === 1. Gerar o QR Code ===
data = "Natália Vieira Lima"
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
qr_h, qr_w = qr_array.shape

# === 2. Carregar imagem base (BOB) ===
#path = os.path.join(base_path, "CN.png")  # imagem na mesma pasta
#path = os.path.join(base_path, "dogs.jpeg")  # imagem na mesma pasta
path = os.path.join(base_path, "bob.jpg")  # imagem na mesma pasta

img = cv2.imread(path)
cv2.imwrite(os.path.join(base_path, "original.png"), img)

h, w, _ = img.shape

# ============================================================
# 🔥 NOVO: CORTAR A IMAGEM DO BOB PARA O TAMANHO DO QR
# ============================================================

# coordenadas do recorte central
x1 = (w - qr_w) // 2
y1 = (h - qr_h) // 2
x2 = x1 + qr_w
y2 = y1 + qr_h

# aplica o corte central
img = img[y1:y2, x1:x2]

# salva imagem cortada
cv2.imwrite(os.path.join(base_path, "bob_cortado.png"), img)

# atualiza dimensões após o crop
h, w, _ = img.shape
print(f"Imagem cortada para: {w}x{h} px")

# ============================================================

# === 3. Separar canais RGB ===
b, g, r = cv2.split(img)

# === 4. Criar máscara ===
qr_norm = qr_array / 255.0
mask = 1 - qr_norm
alpha = 1

# === 5. Função para aplicar QR em um canal ===
def aplicar_qr(canal):
    canal_mod = canal.copy().astype(np.float32)
    # agora o QR ocupa a imagem inteira (0,0)
    roi = canal_mod[0:qr_h, 0:qr_w]
    roi_new = roi * (1.0 - alpha * mask)
    canal_mod[0:qr_h, 0:qr_w] = np.clip(roi_new, 0, 255)
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
cv2.imshow("Original Cortada", img)
cv2.imshow("QR Gray", qr_array)
cv2.imshow("Modulado R", img_R)
cv2.imshow("Modulado G", img_G)
cv2.imshow("Modulado B", img_B)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Todas as imagens foram salvas em:\n{base_path}")
