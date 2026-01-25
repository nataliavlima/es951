import qrcode
import qrcode.constants
import numpy as np
import cv2
import os

data = "https://www.fem.unicamp.br/"

versions = [8, 10, 12]

error_corrections = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}

base_path = "qr_referencia"
os.makedirs(base_path, exist_ok=True)

for version in versions:
    for ec_label, ec_value in error_corrections.items():
        try:
            qr = qrcode.QRCode(
                version=version,
                error_correction=ec_value,
                box_size=10,
                border=1,
            )

            qr.add_data(data)
            qr.make(fit=False)

            qr_img = qr.make_image(
                fill_color="black",
                back_color="white"
            ).convert("L")

            qr_array = np.array(qr_img)

            filename = f"qr_v{version}_ec{ec_label}.png"
            filepath = os.path.join(base_path, filename)

            cv2.imwrite(filepath, qr_array)
            print(f"Gerado com sucesso: {filename}")

        except Exception as e:
            print(f"Falha ao gerar QR v{version} EC-{ec_label}: {e}")
