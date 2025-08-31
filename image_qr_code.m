clc; close all force; clear all;

bob = imread('bob.png'); % converte imagem em binario
imshow(bob);              % mostra como ficou


% RGB <-> HSV
hsv_bob = rgb2hsv(bob);
% imshow(hsv_bob);

rgb_bob = hsv2rgb(hsv_bob);
% imshow(rgb_bob);

% Gray Scale e Binarizacao
bob_gray =  rgb2gray(rgb_bob);
imshow(bob_gray);
BW = imbinarize(bob_gray); % binariza a imagem Otsu's method

% figure
% imshowpair(bob,BW,'montage');

% repetir com o QR
qr = imread('QR.jpg');
qr_gray =  rgb2gray(qr);
imshow(qr_gray);
qr_BW = imbinarize(qr_gray); % binariza a imagem Otsu's method
figure
imshowpair(qr,qr_BW,'montage');