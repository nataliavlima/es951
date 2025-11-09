import cv2
import numpy as np
import qrcode
import os

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

# Converter para escala de cinza (L = luminância 0–255)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("L")
qr_array = np.array(qr_img)

# Redimensionar QR para 200×200
qr_array = cv2.resize(qr_array, (200, 200), interpolation=cv2.INTER_AREA)

# === 2. Carregar imagem base ===
path = r"C:\Users\natalia vieira lima\OneDrive\Documentos\1. Unicamp\es951\Embed_qr_code\RGB_manipulation\bob.jpg"
print("Existe o arquivo?", os.path.exists(path))

img = cv2.imread(path)
h, w, _ = img.shape
qh, qw = qr_array.shape

# === 3. Calcular posição central do QR ===
x_offset = (w - qw) // 2
y_offset = (h - qh) // 2

# === 4. Separar canais RGB ===
b, g, r = cv2.split(img)

# === 5. Criar máscara normalizada ===
qr_norm = qr_array / 255.0        # 0 (preto) → 0.0, 255 (branco) → 1.0
mask = 1 - qr_norm                # Inverte: preto → 1, branco → 0

# === 6. Modificar apenas o canal R ===
alpha = 1.0                       # intensidade da modulação

# Converter para float para evitar saturação
r_mod = r.astype(np.float32)

# Selecionar apenas a região onde o QR será aplicado
roi = r_mod[y_offset:y_offset + qh, x_offset:x_offset + qw]

# Aplicar a modulação — escurece onde o QR é preto
roi_new = roi * (1.0 - alpha * mask)

# Reatribuir a ROI modificada
r_mod[y_offset:y_offset + qh, x_offset:x_offset + qw] = np.clip(roi_new, 0, 255)
r_mod = r_mod.astype(np.uint8)

# === 7. Recompor e salvar ===
img_mod = cv2.merge((b, g, r_mod))
cv2.imwrite("saida_qr_modulado.png", img_mod)

# === 8. Mostrar resultado ===
cv2.imshow("QR Code", qr_array)
cv2.imshow("Imagem Original", img)
cv2.imshow("Imagem com QR no Canal R", img_mod)
cv2.waitKey(0)
cv2.destroyAllWindows()
