function [XP,YP,ixrp,iyrp, xrp,yrp,all_C,A,B,rf,shift,scale ] = hypercube_data(...
    num_class, num_useful_feat, num_clust_per_class, num_pat_per_cluster, ...
    num_redundant_feat, num_repeat_feat, num_useless_feat, class_sep, ...
    flip_y, num_repeat_val, rnd, debug, xrp, yrp, all_C, A, B, rf, shift, scale)
%[XP,YP,ixrp,iyrp, xrp,yrp,all_C,A,B,rf,shift,scale ] = hypercube_data(
% num_class 1, num_useful_feat 2, num_clust_per_class 3, num_pat_per_cluster 4, 
% num_redundant_feat 5, num_repeat_feat 6, num_useless_feat 7, class_sep 8, 
% flip_y 9, num_repeat_val 10, rnd 11, debug 12, xrp 13, yrp 14, all_C 15, 
% A 16, B 17,rf 18,shift 19,scale 20)
%
% Draws a pattern recognition problem at random, for a num_class-class problem.
%
% Useful features:
% Each class is composed of a number of Gaussian clusters that are on the
% vertices of a hypercube in a subspace of dimension num_useful_feat.
% N(0,1) is used to draw the examples of independent features for each cluster.
% Some covariance is added by multiplying by a random matrix A,
% with uniformly distributed random numbers between -1 and 1.
% The clusters are then placed on the hypercube vertices.
% The hypercube vertices are placed at values +-class_sep.
%
% Redundant features:
% Useful features are multiplied by a random matrix B,
% with uniformly distributed random numbers between -1 and 1.
%
% Repeated features:
% Drawn randomly from useful and redundant features.
%
% Useless features:
% Additional features drawn at random not related to the concept.
% Features are then shifted and rescaled randomly to span 3 orders of magnitude.
% Random noise is then added to the features according to N(0,.1) to create several replicates.
% if flip_y is provided, a random fraction flip_y of labels are randomly exchanged.
%
% -- Aknowledgements: The idea is inspired by the work of Simon Perkins.
% Inputs:
% num_class -- Number of classes
% num_useful_feat -- Number of features initially drawn to explain the concept
% num_clust_per_class -- Number of cluster per class
% num_pat_per_cluster -- Number of patterns per cluster // all balanced for now, can be
% generalized to imbalanced classes (can take subset of samples of each class)
% num_redundant_feat -- Number of features linearly dependent upon the useful features
% num_repeat_feat -- Number of features repeating the previous ones (drawn at random)
% num_useless_feat -- Number of features dran at random regardless of class label information
% class_sep -- Factor multiplying the hypercube dimension.
% flip_y -- Fraction of y labels to be randomly exchanged.
% num_repeat_val -- number of times each entry is repeated (modulo some noise).
% rnd -- Flag to enable or disable random permutations.
% debug -- 0/1 flag.
% Returns:
% XP -- Matrix (num_pat, num_feat, num_repeat_val) of randomly permuted features
% YP -- Vector of 0,1...num_class target class labels (in random order, to be 
% used eventually for clustering)
% ixrp -- permutation matrix to be used to restore the original feature order
% iyrp -- permutation matrix to be used to restore the original pattern order
% (class labels of the same class are consecutive
% and there are the same number of example per class, before label corruption)
% Y=YP(iyrp); X=XP(iyrp,ixrp);
% all_C -- A matrix 2^num_useful_feat*num_useful_feat of
% hypercube vertices where to place the cluter centers.
% A -- Matrix used to correlate the useful features.
% B -- Matrix used to create dependent (redundant) features.
% rf -- Indices of repeated features.
% shift -- Shift applied.
% scale -- Scale applied.
% Isabelle Guyon -- July 2003 -- isabelle@clopinet.com

if nargin<8, class_sep=1; end
if nargin<9, flip_y=0; end
if nargin<10, num_repeat_val=1; end
if nargin<11, rnd=0; end % disable random permutation
if nargin<12, debug=0; end
if nargin<13, xrp=[]; end
if nargin<14, yrp=[]; end
if nargin<15, all_C=[]; end
if nargin<16, A={}; end
if nargin<17, B=[]; end
if nargin<18, rf=[]; end
if nargin<19, shift=[]; end
if nargin<20, scale=[]; end

% Count features and patterns
num_feat=num_useful_feat + num_repeat_feat + num_redundant_feat + num_useless_feat;
num_pat_per_class=num_pat_per_cluster*num_clust_per_class;
num_pat=num_pat_per_class*num_class;

X=zeros(num_pat, num_feat);
% Attribute class labels
y=0:num_class-1;
Y=repmat(y, num_pat_per_class, 1);
Y=Y(:);

% Hypercube design
is_XOR=0;
if num_useful_feat==2 & num_class==2 & num_clust_per_class==2, 
    is_XOR=1;
 all_C=[-1 -1; 1 1; 1 -1; -1 1]; % XOR
else
 if isempty(all_C)
 fprintf('New C\n');
 all_C=2*ff2n(num_useful_feat)-1;
 rndidx=randperm(size(all_C,1));
 all_C=all_C(rndidx,:);
 end
end
% Draw A
if isempty(A)
 fprintf('New A\n');
 for k=1:num_class*num_clust_per_class
 A{k} = 2*rand(num_useful_feat, num_useful_feat)-1;
 end
end
% Loop over all clusters
for k=1:num_class*num_clust_per_class
 % define the range of patterns of that cluster
 kmin=(k-1)*num_pat_per_cluster+1;
 kmax=kmin+num_pat_per_cluster-1;
 kidx=kmin:kmax;
 % Draw n features independently at random
 X(kidx,1:num_useful_feat)=random('norm', 0, 1, num_pat_per_cluster,num_useful_feat);
 % Multiply by a random matrix to create some co-variance of the features
 X(kidx,1:num_useful_feat)=X(kidx,1:num_useful_feat)*A{k};
 % Shift the center off zero to separate the clusters
 C=all_C(k,:)*class_sep;
 X(kidx,1:num_useful_feat) = X(kidx,1:num_useful_feat) + repmat(C,num_pat_per_cluster, 1);
end
if debug,
 featdisplay(normalize_data([X(:,1:num_useful_feat),Y])); title('Useful features');
 figure; scatterplot(X(:, 1:num_useful_feat), Y); title('Useful features');
end
% Create redundant features by multiplying by a random matrix
if isempty(B),
 fprintf('New B\n');
 B = 2*rand(num_useful_feat, num_redundant_feat)-1;
end
X(:,num_useful_feat+1:num_useful_feat+num_redundant_feat)=X(:,1:num_useful_feat)*B;
if debug,
 featdisplay(normalize_data([X(:,1:num_useful_feat+num_redundant_feat),Y]));
title('Useful+redundant features');
 figure; scatterplot(X(:, 1:num_useful_feat+num_redundant_feat), Y);
title('Useful+redundant features');
end
% Repeat num_repeat_feat features, chosen at random among useful and redundant feat
nf=num_useful_feat+num_redundant_feat;
if isempty(rf)
 fprintf('New rf\n');
 rf=round(1+rand(num_repeat_feat,1)*(nf-1));
end
X(:,nf+1:nf+num_repeat_feat)=X(:,rf);
if debug,
featdisplay(normalize_data([X(:,1:num_useful_feat+num_redundant_feat+num_repeat_feat),Y]));
 title('Useful+redundant+repeated features');
end
% Add useless features : these are uncorrelated with one another, but could be correlated :=)
X(:,num_feat-num_useless_feat+1:num_feat)=random('norm', 0, 1, num_pat, num_useless_feat);
if debug,
 featdisplay(normalize_data([X,Y]));
 title('All features');
end
% Add random y label errors
num_err_pat = round(num_pat*flip_y);
rp=randperm(num_pat);
fi=rp(1:num_err_pat);
Y(fi)=mod(Y(fi)+round(rand(num_err_pat,1)*(num_class-1)), num_class);
if debug,
 featdisplay(normalize_data([X,Y]));
 title('All features + flipped labels');
end
% Randomly shift and scale
if isempty(shift)
 fprintf('New shift\n');
 shift=rand(num_feat,1);
end
if isempty(scale)
 fprintf('New scale\n');
 scale=1+100*rand(num_feat,1);
end
X=X+repmat(shift',num_pat,1);
X=X.*repmat(scale',num_pat,1);
if debug,
 featdisplay([X,100*normalize_data(Y)]);
 title('All features + flipped labels + scale shifted');
end
% Randomly permute the features and patterns
if isempty(xrp)
 fprintf('New xrp, yrp\n');
 if rnd
 xrp=randperm(num_feat);
 yrp=randperm(num_pat);
else
 xrp=1:num_feat;
 yrp=1:num_pat;
end
end
XP0=X(yrp,xrp);
YP=Y(yrp);
if debug,
 [ys,pattidx]=sort(YP);
 featdisplay(normalize_data([XP0(pattidx,:),YP(pattidx)]));
 title('After permutation and data normalization');
end
% Create inverse random indices
ixrp(xrp)=1:num_feat;
iyrp(yrp)=1:num_pat;
% Create several replicates by adding a little bit of random noise
XP=zeros(num_pat, num_feat, num_repeat_val);
for k=1:num_repeat_val
 N=random('norm', 0, .1*sqrt(num_repeat_val), num_pat, num_feat);
 XP(:,:,k)=XP0.*(1+N);
end
if debug,
 featdisplay(normalize_data([XP(pattidx,:),YP(pattidx)]));
 title('After adding noise');
end
