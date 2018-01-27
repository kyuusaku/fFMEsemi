function [aIdx, val, elapsed_time] = find_idx(anchor, X_train)

t_start = tic;

D = sqdist(anchor, X_train);
[val, pos] = sort(D, 2);
aIdx = pos(:,1);
val = val(:,1);

elapsed_time = toc(t_start);