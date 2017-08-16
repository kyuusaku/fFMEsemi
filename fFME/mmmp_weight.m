function [Z, elapsed_time] = mmmp_weight(E, anchor, aIdx, X_train, k)

if isempty(aIdx)
    % obtain aIdx
    [index, search_params] = flann_build_index(X_train, ...
        struct('algorithm', 'kdtree', 'trees', 8, 'checks', 128));
    search_params.cores = 0;
    aIdx = flann_search(index, anchor, 1, search_params);
end

t_start = tic;
% 
n = size(X_train, 2);
m = size(anchor, 2);

Z = zeros(n, m); % init output weight

y = zeros(n, 1); % init y
y(aIdx) = 1:m; 
% test a=zeros(1,5) a([2,3,4])=1:3 a=[0 1 2 3 0] a([2,4,3])=1:3 a=[0 1 3 2
% 0]
	
d = inf(n, 1);
d(aIdx) = 0;
    
mask = zeros(n, k);
    
for i = 1:k
    y_pred = y;
    [num_iter, size_Q, iters] = masked_mmlp_core(aIdx, E, y_pred, d, mask, 0.9999);
    Z(:, y_pred) = d;
    mask(:, i) = y_pred;
    mask(:, aIdx) = 0;
end

Z = Z ./ (max(Z, 2) * ones(1,m));
Z = exp(-Z);
Z = Z ./ (sum(Z, 2) * ones(1,m));

Z=sparse(Z);

elapsed_time = toc(t_start);