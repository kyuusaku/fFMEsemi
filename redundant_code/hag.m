function [ZH, Z01, zh_time, rLh, rLh_time] = hag(X_train, anchors, s)
% X_train(dXn): input data matrix, d: dimension, n: # samples
% s: # of closest anchors

n_hierarchy = numel(anchors);
zh_time = zeros(n_hierarchy, 1);

[Z01, zh_time(1)] = flann_WeightEstimation(X_train, anchors{1}, s);
ZH = Z01;
for i = 2 : n_hierarchy
    [Z, zh_time(i)] = flann_WeightEstimation(anchors{i-1}, anchors{i}, s);
    ZH = ZH * Z;
end
clear Z;
zh_time = sum(zh_time);

tic;
Z01ZH = Z01' * ZH;
rLh = ZH'*ZH - Z01ZH'*diag(sum(Z01).^-1)*Z01ZH;
rLh_time = toc;
 