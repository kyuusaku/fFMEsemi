function [W, b, F] = aFME_semi(X, Z, rLz, Y, para)
% INPUT
%     X: f*m matrix, each colomn is a data point
%     Z: n*m matrix, sample adaptive weights
%     rLz: m*m matrix, reduced normalized graph
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

[dim,m] = size(X);

Xc = bsxfun(@minus, X, mean(X,2));
W = (Xc * Xc' + para.gamma .* eye(dim)) \ Xc;

n = size(Y,1);
u = para.uu .* ones(n,1);
u(sum(Y,2) == 1) = para.ul;
U = spdiags(u,0,n,n);

A = (rLz + Z'*U*Z + para.mu*eye(m) - (para.mu/m)*ones(m,1)*ones(1,m) ...
    - para.mu*Xc'*W) \ (Z'*U*Y); % A is a m*c matrix

F = Z*A;
W = W*A;
b = 1/m*(sum(A,1)' - W'*(X*ones(m,1)));