function [U, M] = pca(X, dim, epsilon, row)
%   [U, M] = pca(X, dim)
%   X : input data, f by n matrix, n is the number of samples,
%        f is the original feature dimension.
%   dim: feature dimension of the output data.
%        also can be 0 to 1, the percent energy of the data would be preserved.
%   epsilon: parameter of whiten.
%   row: indicates per-example mean subtraction (remove DC).
%   M: the mean of input data
%   U: the projection matrix

%% #dim & #samples
[nFea, nSmp] = size(X);

%% feature normalize
if exist('row', 'var') && row == 1
    M = mean(X, 1);
else
    M = mean(X, 2);
end
X = bsxfun(@minus, X, M);

%% compute pca
if nFea < nSmp
    C = (X * X') ./ nSmp;
else
    C = (X' * X) ./ nSmp;
end
C(isnan(C)) = 0;
C(isinf(C)) = 0;
[U, S] = eig(C);
if ~(nFea < nSmp)
    U = X * U;
end
S = diag(S);
[S, index] = sort(S, 'descend');

%% project data
if dim == 0
    dim = nFea;
elseif dim < 1
    dim = find(cumsum(S ./ sum(S)) >= dim, 1, 'first');
end
U = U(:,index(1:dim));
S = S(1:dim);
if exist('epsilon', 'var') && ~isempty(epsilon)
    U = U * diag(1./sqrt(abs(S) + epsilon));
end