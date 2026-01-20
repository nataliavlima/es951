import cv2
import numpy as np
import qrcode
import os

# ======================
# Parâmetros
# ======================
texto_qr = "Incorporação híbrida RGB + HSV"
alpha_rgb = 0.5
alpha_hsv = 0.4
beta = 0.6
saida_dir = os.path.join(os.path.dirname(__file__), "Hybrid")
os.makedirs(saida_dir, exist_ok=True)

# ======================
# Carregar imagem base
# ======================
#path = r"C:\Users\natalia vieira lima\OneDrive\Documentos\1. Unicamp\es951\Embed_qr_code\HSV\bob.jpg"
#path = r"C:\Users\natalia vieira lima\OneDrive\Documentos\1. Unicamp\es951\Embed_qr_code\HSV\CN.png"
path = r"C:\Users\natalia vieira lima\OneDrive\Documentos\1. Unicamp\es951\Embed_qr_code\HSV\FERB.jpg"

img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Não foi possível ler a imagem em {path}")

print("Imagem carregada:", img.shape)

# ======================
# Gerar QR Code dimensionado
# ======================
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

# Redimensionar QR para 200×200
qr_array = cv2.resize(qr_array, (200, 200), interpolation=cv2.INTER_AREA)

# ======================
# Redimensionar imagem ao tamanho do QR
# ======================
img = cv2.resize(img, (qr_array.shape[1], qr_array.shape[0]), interpolation=cv2.INTER_AREA)

h, w, _ = img.shape
print("Imagem redimensionada para:", img.shape)

# ======================
# Centralizar QR
# ======================
x_offset = (w - qr_array.shape[1]) // 2
y_offset = (h - qr_array.shape[0]) // 2

mask = 1 - (qr_array / 255.0)

print(f"QR Code tamanho: {qr_array.shape}, posição: ({x_offset}, {y_offset})")

# ======================
# Modulação RGB (canal R)
# ======================
b, g, r = cv2.split(img)
r_mod = g.astype(np.float32)

roi_r = r_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]]
roi_mask = mask[:roi_r.shape[0], :roi_r.shape[1]]
roi_r_new = roi_r * (1.0 - alpha_rgb * roi_mask)

r_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]] = np.clip(roi_r_new, 0, 255)

img_rgb_mod = cv2.merge([b, g, r_mod.astype(np.uint8)])
cv2.imwrite(os.path.join(saida_dir, "saida_rgb.png"), img_rgb_mod)

# ======================
# Modulação HSV (canal V)
# ======================
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_ch, s_ch, v_ch = cv2.split(img_hsv)
v_mod = v_ch.astype(np.float32)

roi_v = v_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]]
roi_mask = mask[:roi_v.shape[0], :roi_v.shape[1]]
roi_v_new = roi_v * (1.0 - alpha_hsv * roi_mask)

v_mod[y_offset:y_offset + qr_array.shape[0], x_offset:x_offset + qr_array.shape[1]] = np.clip(roi_v_new, 0, 255)

img_hsv_mod = cv2.merge([h_ch, s_ch, v_mod.astype(np.uint8)])
img_hsv_mod_bgr = cv2.cvtColor(img_hsv_mod, cv2.COLOR_HSV2BGR)
cv2.imwrite(os.path.join(saida_dir, "saida_hsv.png"), img_hsv_mod_bgr)

# ======================
# Fusão híbrida RGB + HSV
# ======================
img_hybrid = cv2.addWeighted(img_rgb_mod, beta, img_hsv_mod_bgr, 1 - beta, 0)
cv2.imwrite(os.path.join(saida_dir, "saida_hibrida.png"), img_hybrid)

# ======================
# Salvar auxiliares
# ======================
cv2.imwrite(os.path.join(saida_dir, "original.png"), img)
cv2.imwrite(os.path.join(saida_dir, "qr_gray.png"), qr_array)

print(f"Todas as imagens foram salvas em: {os.path.abspath(saida_dir)}")
