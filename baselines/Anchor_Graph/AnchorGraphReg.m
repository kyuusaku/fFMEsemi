function [F, A, err, elapsed_time] = AnchorGraphReg(Z, rL, ground, label_index, gamma, class_norm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% AnchorGraphReg 
% Written by Wei Liu (wliu@ee.columbia.edu)
% Z(nXm): regression weight matrix
% rL(mXm): reduced graph Laplacian 
% ground(1Xn): [1 1 1 2 2 2] discrete groundtruth class labels 
% label_index(1Xln):  given index of labeled data
% gamma: regularization parameter, set to 0.001-1
% F(nXC): soft label scores on raw data
% A(mXC): soft label scores on anchors
% err: classification error rate on unlabeled data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

[n,m] = size(Z);
ln = length(label_index);
C = max(ground);

Yl = zeros(ln,C);
for i = 1:C
    ind = find(ground(label_index) == i);
    Yl(ind',i) = 1;
%     clear ind;
end

Zl = Z(label_index',:);
LM = Zl'*Zl+gamma*rL; 
% clear rL; 
RM = Zl'*Yl;  
% clear Yl;
% clear Zl;
A = (LM+1e-6*eye(m))\RM; 
% clear LM;
% clear RM;

F = Z*A; 
% clear Z;
% tmpF = sum(F)';
% F1 = F*spdiags(tmpF.^-1, 0, numel(tmpF), numel(tmpF)); clear temF;
% if size(F,2) < 2000
if class_norm
    F1 = F*diag(sum(F).^-1);
else
    F1 = F;
end
    elapsed_time = toc;
    [temp,order] = max(F1,[],2);
% else
%     for i = 1 : size(F, 2)
%         F(:,i) = F(:,i) ./ sum(F(:,i));
%     end
%     [temp,order] = max(F,[],2);
% end

% clear temp;
% clear F1;
output = order';
% clear order;
output(label_index) = ground(label_index);
err = length(find(output ~= ground))/(n-ln);
% clear output;

