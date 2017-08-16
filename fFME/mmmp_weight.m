function [Z, elapsed_time, D] = mmmp_weight(E, anchor, aIdx, X_train, k)

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

D = zeros(n, m); % init minmax distance matrix

parfor i = 1 : m
    y_pred = zeros(n, 1);
    y_pred(aIdx(i)) = i;
    
    d = inf(n, 1);
    d(aIdx(i)) = 0;

    [num_iter, size_Q, iters] = mmlp_core(aIdx(i), E, y_pred, d, 0.9999);
    
    D(:, i) = d;
end

[val, pos] = sort(D, 2);

val = val(:,1:k);
pos = pos(:,1:k);

val = val ./ (max(val, 2) * ones(1,k));
val = exp(-val);
val = val ./ (sum(val, 2) * ones(1,k));

Z = sparse(reshape(repmat(1:n, k,1), 1,[]), reshape(pos', 1,[]), reshape(val', 1,[]), n, m);

elapsed_time = toc(t_start);