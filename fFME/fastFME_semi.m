function [W, b, F] = fastFME_semi(X, B, Y, para, class_norm)
% INPUT
%     X: f*n matrix, each colomn is a data point
%     B: n*m matrix, sample adaptive weights
%     Y: n*c matrix, class indicator matrix. 
%        Yij=1 if xi is labeled as j, Yij=0 otherwise
%     para: parameters
%       para.ul: parameter associated to labeled data points
%       para.uu: parameter associated to unlabeled data points(usually be 0)
%       para.mu: the parameter
%       para.gamma: the parameter
% OUTPUT
%     W: f*c projection matrix
%     b: c*1 bias vector
%     F: n*c soft label matrix

[dim,n] = size(X);

Xc = bsxfun(@minus, X, mean(X,2));
A = Xc * Xc' + para.gamma .* eye(dim);

u = para.uu .* ones(n,1);
u(sum(Y,2) == 1) = para.ul;
U = spdiags(u,0,n,n);

V_inv = spdiags((u + (para.mu + 1) .* ones(n,1)).^-1, 0, n, n);
if dim > 50
    G = [full(B) ones(n,1) Xc'];
else
    G = [B ones(n,1) Xc'];
end
Sigma = diag(sum(B));
M_inv = blkdiag(Sigma, n/para.mu, A ./ para.mu);
VUY = V_inv * U * Y;

F = VUY + ...
    V_inv * (G * (full(M_inv - G' * V_inv * G) \ (G' * VUY)));
W = A \ (Xc * F);
b = 1/n*(sum(F,1)' - W'*(X*ones(n,1)));

if class_norm
    F = F * diag(sum(F).^-1);
end