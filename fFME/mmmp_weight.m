function [Z, elapsed_time, D] = mmmp_weight(E, aIdx, k)

t_start = tic;
% 
n = size(E, 1);
m = numel(aIdx);

D = zeros(n, m); % init minmax distance matrix

parfor i = 1 : m
    y_pred = zeros(n, 1);
    y_pred(aIdx(i)) = i;
    
    d = inf(n, 1);
    d(aIdx(i)) = 0;

    [num_iter, size_Q, iters] = my_mmlp_core(aIdx(i), E, y_pred, d);%, 0.9999);
%     fprintf('%d/%d num_iter:%d\n', i, m, num_iter);
    
    D(:, i) = d;
end

[val, pos] = sort(D, 2);

val = val(:,1:k);
pos = pos(:,1:k);

val = val ./ (max(val, [], 2) * ones(1,k));
val = exp(-val);
val(isnan(val)) = 0;
val = val ./ (sum(val, 2) * ones(1,k));

Z = sparse(reshape(repmat(1:n, k,1), 1,[]), reshape(pos', 1,[]), reshape(val', 1,[]), n, m);

elapsed_time = toc(t_start);