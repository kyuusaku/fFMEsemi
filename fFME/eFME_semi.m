function [W, b, F] = eFME_semi(X, Z, rLz, Y, para)
% INPUT
%     X: f*n matrix, each colomn is a data point
%     Z: n*m matrix, sample adaptive weights
%     rLz: reduced 
%     Y: n*c matrix, class indicator matrix. 
%        Yij=1 if xi is labeled as j, Yij=0 otherwise
%     para: parameters
%       para.ul: parameter associated to labeled data points
%       para.uu: parameter associated to unlabeled data points(usually be 0)
%       para.mu: the parameter
%       para.gamma: the parameter
% OUTPUT
%     W: projection matrix
%     b: bias vector
%     F: soft label matrix

[dim,n] = size(X);

Xc = bsxfun(@minus, X, mean(X,2));
XcZ = Xc * Z;
W = (Xc * Xc' + para.gamma .* eye(dim)) \ XcZ;

u = para.uu .* ones(n,1);
u(sum(Y,2) == 1) = para.ul;
U = spdiags(u,0,n,n);
UI = spdiags((u + para.mu),0,n,n);

Z1 = ones(1,n) * Z;

A = (rLz + Z'*UI*Z - (para.mu/n)*(Z1')*Z1 - para.mu*XcZ'*W) \ (Z'*U*Y);

F = Z*A;
W = W*A;
b = 1/n*(sum(F,1)' - W'*(X*ones(n,1)));