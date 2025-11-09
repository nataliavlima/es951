import cv2
import numpy as np
import qrcode

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

# Redimensionar o QR para caber na imagem base
qr_array = cv2.resize(qr_array, (200, 200), interpolation=cv2.INTER_AREA)

# === 2. Carregar imagem base ===
import os

path = r"C:\Users\natalia vieira lima\OneDrive\Documentos\1. Unicamp\es951\Embed_qr_code\RGB_manipulation\bob.jpg"
print("Existe o arquivo?", os.path.exists(path))

img = cv2.imread(path)
h, w, _ = img.shape

# Centralizar o QR na imagem
x_offset = (w - qr_array.shape[1]) // 2
y_offset = (h - qr_array.shape[0]) // 2

# === 3. Separar os canais RGB ===
b, g, r = cv2.split(img)

# === 4. Modificar apenas um canal (ex: R) ===
# Normalizar QR para [0, 1] e inverter (preto = 1)
mask = 1 - (qr_array / 255.0)

# Ajuste de intensidade da modulação
alpha = 1  # força da incorporação (0 = nada, 1 = totalmente QR)

# Criar uma cópia do canal R
r_mod = r.copy()

# Inserir QR centralizado
r_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]] = \
    (1 - alpha) * r[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]] + \
    alpha * (r[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]] * mask)

# === 5. Recompor e salvar ===
img_mod = cv2.merge((b, g, r_mod))
cv2.imwrite("saida_qr_modulado.png", img_mod)

# Mostrar resultado
cv2.imshow("QR", qr_array)
cv2.imshow("Original", img)
cv2.imshow("Modulado (canal R)", img_mod)
cv2.waitKey(0)
cv2.destroyAllWindows()
