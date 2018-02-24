function [W, b] = LapRLS(X_train, Y_train, L, label_ind, gammaA, gammaI)
nFea = size(X_train, 1);
% construct the labeled matrix
feaLabel = X_train(:, label_ind);
gndLabel = Y_train(label_ind);
classLabel = unique(gndLabel);
nClass = numel(classLabel);
nLabel = numel(gndLabel);
YLabel = zeros(nLabel, nClass);
for i = 1 : nClass
    YLabel(gndLabel == i, classLabel(i)) = 1;
end
% compute W
Xl = bsxfun(@minus, feaLabel, mean(feaLabel,2));
W = (Xl * Xl' + gammaA * nLabel .* eye(nFea) + ...
    gammaI * nLabel .* (X_train * L * X_train')) \ (Xl * YLabel);
b = 1/nLabel*(sum(YLabel,1)' - W'*(feaLabel*ones(nLabel,1)));

end