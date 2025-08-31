% QR-Images (versão prática) — MATLAB demo
% Requer: Image Processing Toolbox (para readBarcode/imbinarize/imresize/etc.)

clear; clc;

% ======= Parâmetros do usuário =======
qrFile            = 'qr.png';     % QR base (preto no branco)
targetFile        = 'target.png'; % Imagem alvo
moduleSize        = 12;           % pixels por módulo (ajuste ao seu PNG)
quietZoneModules  = 4;            % quiet zone do seu QR (tipicamente 4)
stopAfterAccepted = inf;          % pode limitar nº de inversões aceitas
% =====================================

% --- Utilitários in-line ---
toGray = @(I) (size(I,3)==3) .* rgb2gray(I) + (size(I,3)~=3) .* I;

% ---------- 1) Ler QR base e converter para grade de módulos ----------
Iq = imread(qrFile);
Iq = uint8(toGray(Iq));
BWq = ~imbinarize(Iq);                 % preto=1, branco=0
BWq = cropQuietZoneByContent(BWq);     % remove quiet zone por conteúdo

% Ajustar para múltiplos exatos de moduleSize e quadrado
[h,w] = size(BWq);
N = floor(min(h,w) / moduleSize);
BWq = BWq(1:N*moduleSize, 1:N*moduleSize);

qrModules = blockproc(BWq, [moduleSize moduleSize], ...
    @(blk) mean(blk.data(:)) > 0.5);
qrModules = logical(qrModules);

% Mensagem original (para validar)
img0 = renderModules(qrModules, moduleSize, quietZoneModules);
[msg0, fmt0] = readBarcode(img0);
if isempty(msg0)
    error('Não consegui decodificar o QR base. Verifique moduleSize/quiet zone.');
end
fprintf('Mensagem original decodificada: "%s" (%s)\n', msg0, fmt0);

% ---------- 2) Preparar imagem alvo em N×N ----------
It = imread(targetFile);
It = im2double(toGray(It));
ItR = imresize(It, [N N], 'bilinear');

% Dither (1-bit) — módulos desejados: 1=preto, 0=branco
desired = dither(ItR);            % retorna lógico
desired = ~desired;               % dither usa 1=branco; invertendo para 1=preto

% ---------- 3) Máscara conservadora de áreas fixas ----------
fixedMask = buildConservativeFixedMask(N); % 1 = FIXO (não mexer)
mutableMask = ~fixedMask;

% Candidatos onde difere da imagem E é mutável
cand = find(mutableMask & (qrModules ~= desired));

% Importância: quão "errado" está frente à imagem (maior = tenta antes)
score = abs(ItR(cand) - double(qrModules(cand))); % distância à aparência desejada
[~, order] = sort(score, 'descend');
cand = cand(order);

% ---------- 4) Otimização gulosa com verificação ----------
acc = 0;
best = qrModules;

for k = 1:numel(cand)
    if acc >= stopAfterAccepted
        break;
    end

    idx = cand(k);
    trial = best;
    trial(idx) = desired(idx);  % tenta aproximar da imagem

    imgTry = renderModules(trial, moduleSize, quietZoneModules);

    [msgTry, fmtTry] = readBarcode(imgTry);

    if ~isempty(msgTry) && strcmp(msgTry, msg0)
        best = trial; acc = acc + 1;
        if mod(acc,50)==0
            fprintf('Aceitas %d inversões...\n', acc);
        end
    end
end

fprintf('Total de inversões aceitas: %d\n', acc);

% ---------- 5) Salvar resultado ----------
outImg = renderModules(best, moduleSize, quietZoneModules);
imwrite(outImg, 'qr_stylized.png');

% Validação final
[msgF, fmtF] = readBarcode(outImg);
if isempty(msgF)
    warning('O QR final não decodificou no MATLAB. Tente reduzir mudanças.');
else
    fprintf('OK! Resultado decodifica como: "%s" (%s)\n', msgF, fmtF);
end

% =================== Funções auxiliares ===================

function BWc = cropQuietZoneByContent(BW)
    % Remove a quiet zone branca detectando o primeiro/último pixel preto
    rows = any(BW, 2);
    cols = any(BW, 1);
    r1 = find(rows, 1, 'first'); r2 = find(rows, 1, 'last');
    c1 = find(cols, 1, 'first'); c2 = find(cols, 1, 'last');
    BWc = BW(r1:r2, c1:c2);
end

function img = renderModules(mods, moduleSize, qzModules)
    % Converte matriz de módulos (1=preto,0=branco) em imagem uint8 com quiet zone
    mods = logical(mods);
    base = kron(mods, ones(moduleSize, moduleSize)); % upsample
    base = uint8(255*(1 - base)); % 0=preto -> 255 branco; invertendo
    qz = uint8(255) * ones(size(base,1)+2*qzModules*moduleSize, ...
                           size(base,2)+2*qzModules*moduleSize, 'uint8');
    r = qzModules*moduleSize;
    qz( (1+r):(r+size(base,1)), (1+r):(r+size(base,2)) ) = base;
    img = qz;
end

function fixed = buildConservativeFixedMask(N)
    % Cria uma máscara FIXA 1=fixo, 0=mutável.
    % Conservadora: protege finder patterns 9x9, linhas de timing e bordas.
    fixed = false(N,N);

    % Finder patterns com separador (aprox. 9x9) nos 3 cantos
    fp = 9; % "finder + separador" ~9 módulos
    fixed(1:fp,            1:fp) = true;           % topo-esq
    fixed(1:fp,        end-fp+1:end) = true;       % topo-dir
    fixed(end-fp+1:end,    1:fp) = true;           % baixo-esq

    % Linhas de timing (linha e coluna ~posição 7 em índices 1-based)
    t = 7;
    fixed(t, :) = true; fixed(:, t) = true;

    % Reservas extras perto das bordas (formato/versão costumam ficar ali)
    b = 2;
    fixed(1:b, :) = true; fixed(end-b+1:end, :) = true;
    fixed(:, 1:b) = true; fixed(:, end-b+1:end) = true;
end
