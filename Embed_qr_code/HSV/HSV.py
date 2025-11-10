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

# Converter o QR para escala de cinza (0–255)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("L")
qr_array = np.array(qr_img)
qr_array = cv2.resize(qr_array, (200, 200), interpolation=cv2.INTER_AREA)

# === 2. Carregar imagem base ===
path = r"C:\Users\natalia vieira lima\OneDrive\Documentos\1. Unicamp\es951\Embed_qr_code\RGB_manipulation\CN.png"
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Não foi possível ler a imagem em {path}")

h, w, _ = img.shape
x_offset = (w - qr_array.shape[1]) // 2
y_offset = (h - qr_array.shape[0]) // 2

# === 3. Converter imagem para HSV ===
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(img_hsv)

# Normalizar QR para [0, 1] e inverter (preto=1, branco=0)
mask = 1 - (qr_array / 255.0)
alpha = 0.7  # intensidade da modulação

def apply_mask(channel, mask, alpha):
    ch = channel.copy().astype(np.float32)
    roi = ch[y_offset:y_offset + mask.shape[0], x_offset:x_offset + mask.shape[1]]
    roi_new = roi * (1 - alpha * mask)
    ch[y_offset:y_offset + mask.shape[0], x_offset:x_offset + mask.shape[1]] = np.clip(roi_new, 0, 255)
    return ch.astype(np.uint8)

# Aplicar QR em H, S e V
h_mod = apply_mask(h_channel, mask, alpha)
s_mod = apply_mask(s_channel, mask, alpha)
v_mod = apply_mask(v_channel, mask, alpha)

# Recriar imagens modificadas e converter de volta para BGR
img_H = cv2.cvtColor(cv2.merge((h_mod, s_channel, v_channel)), cv2.COLOR_HSV2BGR)
img_S = cv2.cvtColor(cv2.merge((h_channel, s_mod, v_channel)), cv2.COLOR_HSV2BGR)
img_V = cv2.cvtColor(cv2.merge((h_channel, s_channel, v_mod)), cv2.COLOR_HSV2BGR)

# === 4. Salvar as imagens na pasta "HSV" ===
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "HSV"))
os.makedirs(output_dir, exist_ok=True)
print("Diretório de saída:", output_dir)

def salvar(nome, imagem):
    caminho = os.path.join(output_dir, nome)
    sucesso = cv2.imwrite(caminho, imagem)
    print(f"Salvando {nome}: {sucesso} ({caminho})")

salvar("original.png", img)
salvar("qr_gray.png", qr_array)
salvar("saida_qr_H.png", img_H)
salvar("saida_qr_S.png", img_S)
salvar("saida_qr_V.png", img_V)

# === 5. Mostrar resultados ===
cv2.imshow("QR", qr_array)
cv2.imshow("Original", img)
cv2.imshow("Canal H modificado", img_H)
cv2.imshow("Canal S modificado", img_S)
cv2.imshow("Canal V modificado", img_V)
cv2.waitKey(0)
cv2.destroyAllWindows()
