function S = constructS(fea, k)
% construct similarity matrix with squared euclidean distance
% use k-nn for sparsity
% fea : d by n matrix, d is the feature dimension, n is the number of
% samples

nSmp = size(fea, 2); % number of samples

maxM = 10*1e9; 
BlockSize = floor(maxM / (nSmp * 3 * 8));

G = zeros(nSmp * (k + 1), 3);

for i = 1:ceil(nSmp / BlockSize)
    if i == ceil(nSmp / BlockSize)
        smpIdx = (i-1) * BlockSize + 1 : nSmp;
        gIdx = (i-1) * BlockSize * (k+1) + 1 : nSmp * (k+1);
    else
        smpIdx = (i-1) * BlockSize + 1 : i * BlockSize;
        gIdx = (i-1) * BlockSize * (k+1) + 1 : i * BlockSize * (k+1);
    end
    
    dist = sqdist(fea(:, smpIdx), fea);
    
    nSmpNow = length(smpIdx);
    dump = zeros(nSmpNow, k + 1);
    idx = dump;
    for j = 1 : k + 1
        [dump(:,j),idx(:,j)] = min(dist, [], 2);
        temp = (idx(:,j) - 1) * nSmpNow + (1 : nSmpNow)';
        dist(temp) = 1e100;
    end

    G(gIdx, 1) = repmat(smpIdx', [k+1, 1]);
    G(gIdx, 2) = idx(:);
    G(gIdx, 3) = dump(:);
end
S = sparse(G(:,1), G(:,2), G(:,3), nSmp, nSmp);
S = S - diag(diag(S));
S = max(S, S');