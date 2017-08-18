function [aIdx, val] = find_idx(anchor, X_train)

D = sqdist(anchor', X_train');
[val, pos] = sort(D, 2);
aIdx = pos(:,1);
val = val(:,1);