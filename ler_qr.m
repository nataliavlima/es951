clc; close all force; clear all;

qrFile            = 'teste.png';     % QR base (preto no branco)

% ---------- Ler QR base ----------
Iq = imread(qrFile);
Iq = rgb2gray(Iq);
BWq = ~imbinarize(Iq);                 % preto=1, branco=0
% BWq = cropQuietZoneByContent(BWq);     % remove quiet zone por conteúdo

% Mensagem original (para validar)
% img0 = renderModules(qrFile, moduleSize, quietZoneModules);
[msg0, fmt0] = readBarcode(Iq);
if isempty(msg0)
    error('Não consegui decodificar o QR base. Verifique moduleSize/quiet zone.');
end
fprintf('Mensagem original decodificada: "%s" (%s)\n', msg0, fmt0);
