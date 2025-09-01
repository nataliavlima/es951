clc; close all force; clear all;

% Parametros
img = imread('bob2.jpg');
qr = imread('teste.png');     % QR base (preto no branco)
qr_text = 'teste.png';

altura_qr = size(qr, 1);
largura_qr = size(qr, 2);

disp("Altura QR: " + altura_qr)
disp("Largura QR: " + largura_qr)

altura_img = size(img, 1);
largura_img = size(img, 2);
disp("Altura IMG: " + altura_img)
disp("Largura IMG: " + largura_img)

% define imagem no tamanho do QR CODE
img = imresize(img, [altura_qr, largura_qr]);
imshow(img);

% Le e transcreve o QR Code
Iq = imread(qr_text);
msg = readBarcode(Iq);
disp("Decoded barcode message: " + msg)

% Gray Scale e Binarizacao
qr_gray =  rgb2gray(qr);
BW = imbinarize(qr_gray); % binariza a imagem Otsu's method
imshow(BW);

% HSV da imagem desejada
img_hsv = rgb2hsv(img);
imshow(img_hsv);