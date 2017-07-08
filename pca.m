function [U, M] = pca(X, dim)

%% #dim & #samples
[nFea, nSmp] = size(X);

M = mean(X,2);
X_center = bsxfun(@minus, X, M);

C = 1/(nSmp-1)*(X_center*X_center');
[U, S] = eig(C);

S = diag(S);
[S, index] = sort(S, 'descend');

if dim == 0
    dim = nFea;
elseif dim < 1
    dim = find(cumsum(S ./ sum(S)) >= dim, 1, 'first');
end

U = U(:,index(1:dim));
%S = S(1:dim);

