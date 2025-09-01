% Exemplo: Misturar QR com uma imagem usando HSV (canal Value)

clear; clc;

% --- 1. Ler imagens ---
qr = imread('qr.png');           % QR base (preto e branco)
img = imread('target.png');      % Imagem alvo (qualquer)

% Redimensionar a imagem para caber no QR
img = imresize(img, [size(qr,1) size(qr,2)]);

% Converter tudo para double (0–1)
qr = im2double(rgb2gray(qr));    % QR é preto e branco -> 1 canal
img = im2double(img);

% --- 2. Converter imagem alvo para HSV ---
img_hsv = rgb2hsv(img);

% --- 3. Modificar o canal V (Value) ---
% Ideia: usar o QR como máscara de intensidade.
% Onde o QR é preto, escurecemos; onde é branco, deixamos o brilho da imagem.
alpha = 0.6;  % peso do QR (ajuste entre estética vs. legibilidade)

newV = (1-alpha) * img_hsv(:,:,3) + alpha * (1-qr);

% Atualizar o canal V
img_hsv_mod = img_hsv;
img_hsv_mod(:,:,3) = newV;

% --- 4. Voltar para RGB ---
stylized = hsv2rgb(img_hsv_mod);

% --- 5. Mostrar resultados ---
figure;
subplot(1,3,1); imshow(qr); title('QR Base');
subplot(1,3,2); imshow(img); title('Imagem alvo');
subplot(1,3,3); imshow(stylized); title('QR + Imagem (HSV Value)');

% --- 6. Salvar ---
imwrite(stylized, 'qr_stylized.png');
