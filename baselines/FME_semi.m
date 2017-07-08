function [W, b, F] = FME_semi(X, L, T, para)
% X: each colomn is a data point
% L: Laplacian matrix, the matrix M in the paper
% T: n*c matrix, class indicator matrix. Tij=1 if xi is labeled as j, Tij=0 otherwise
% para: parameters
%       para.ul: parameter associated to labeled data points
%       para.uu: parameter associated to unlabeled data points(usually be 0)
%       para.mu: the parameter \mu in the paper
%       para.lamda: the parameter \gamma in the paper
% W: projection matrix
% b: bias vector
% F: soft label matrix

% Ref: Feiping Nie, Dong Xu, Ivor W. Tsang, Changshui Zhang.
%      Flexible Manifold Embedding: A Framework for Semi-supervised and Unsupervised Dimension Reduction}.
%      Accepted by IEEE Transactions on Image Processing (TIP)



[dim,n] = size(X);

labeled_idx = sum(T,2) == 1;

Xm = mean(X,2);
Xc = X - Xm*ones(1,n);

if dim < n
    St = Xc*Xc';
    A = para.lamda*inv(para.lamda*St+eye(dim))*Xc;
else
    K = Xc'*Xc;
    A = para.lamda*Xc*inv(para.lamda*K+eye(n));
end;

u = para.uu*ones(n,1);
u(labeled_idx) = para.ul;
U = spdiags(u,0,n,n);

Lc = eye(n) - 1/n*ones(n);
M = U + L + para.mu*para.lamda*Lc - para.mu*para.lamda*(Xc'*A);
F = M\(U*T);
W = A*F;
b = 1/n*(sum(F,1)' - W'*X*ones(n,1));
