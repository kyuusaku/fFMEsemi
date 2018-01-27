function [Z, rL, elapsed_time] = AnchorGraph(TrainData, Anchor, s, flag, cn, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% AnchorGraph
% Written by Wei Liu (wliu@ee.columbia.edu)
% TrainData(dXn): input data matrix, d: dimension, n: # samples
% Anchor(dXm): anchor matrix, m: # anchors 
% s: # of closest anchors, usually set to 2-10 
% flag: 0 gives a Gaussian kernel-defined Z and 1 gives a LAE-optimized Z
% cn: # of iterations for LAE, usually set to 5-20; if flag=0, input 'cn' any value
% Z(nXm): output anchor-to-data regression weight matrix 
% rL(mXm): reduced graph Laplacian matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('verbose', 'var') || isempty(verbose)
    verbose = false;
end

[d,m] = size(Anchor);
n = size(TrainData,2);

if verbose
	disp(['### Anchor graph construction with ' num2str(n) ' data points and ' num2str(m) ' anchor points ###']);
end

t_start = tic;

%% Find s nearest anchor points
tic;

Dis = sqdist(TrainData,Anchor);
val = zeros(n,s);
pos = val;
for i = 1:s
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]';
    Dis(tep) = 1e60;
end
clear Dis;
clear tep;
ind = (pos-1)*n+repmat([1:n]',1,s);

if verbose
	disp(['  ' num2str(s) '-nearest anchor points found...' num2str(toc)]);
end

%% Anchor graph construction
if flag == 0
   %% kernel-defined weights
    %% adaptive kernel width I used in ICML'10
    % val = val./repmat(val(:,s),1,s);  
    % val = exp(-val);
    %% unified kernel width that could be better
    sigma = mean(val(:,s).^0.5);
    val = exp(-val/(1/1*sigma^2));
    
    val = repmat(sum(val,2).^-1,1,s).*val;  
else
   %% LAE-optimized weights 
    tic;
    parfor i = 1:n
        x = TrainData(:,i); 
        x = x/norm(x,2);
        U = Anchor(:,pos(i,:));  
        U = U*diag(sum(U.^2).^-0.5);
        val(i,:) = LAE(x,U,cn);
    end
    clear x;
    clear U;
end

Z = sparse(reshape(repmat(1:n, s,1), 1,[]), reshape(pos', 1,[]), reshape(val', 1,[]), n, m);
% Z([ind]) = [val];
% Z = sparse(Z);
clear val;
clear pos;
clear ind;
clear TrainData;
clear Anchor;

T = Z'*Z;
rL = T-T*diag(sum(Z).^-1)*T;
clear T;

%% Measure time
elapsed_time = toc(t_start);

if verbose
	disp([' Elapsed time = ' num2str(elapsed_time)]);
end


function z = LAE(x,U,cn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% LAE (Local Anchor Embedding)
% Written by Wei Liu (wliu@ee.columbia.edu)
% x(dX1): input data vector 
% U(dXs): anchor data matrix, s: the number of closest anchors 
% cn: the number of iterations, 5-20
% z: the s-dimensional coefficient vector   
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[d,s] = size(U);
z0 = ones(s,1)/s; %(U'*U+1e-6*eye(s))\(U'*x); % % %
z1 = z0; 
delta = zeros(1,cn+2);
delta(1) = 0;
delta(2) = 1;
beta = zeros(1,cn+1);
beta(1) = 1;

for t = 1:cn
	alpha = (delta(t)-1)/delta(t+1);
	v = z1+alpha*(z1-z0); %% probe point

	dif = x-U*v;
	gv =  dif'*dif/2;
	clear dif;
	dgv = U'*U*v-U'*x;
	%% seek beta
	for j = 0:100
		b = 2^j*beta(t);
		z = SimplexPr(v-dgv/b);
		dif = x-U*z;
		gz = dif'*dif/2;
		clear dif;
		dif = z-v;
		gvz = gv+dgv'*dif+b*dif'*dif/2;
		clear dif;
		if gz <= gvz
			beta(t+1) = b;
			z0 = z1;
			z1 = z;
			break;
		end
	end
	if beta(t+1) == 0
		beta(t+1) = b;
		z0 = z1;
		z1 = z;
	end
	clear z;
	clear dgv;
	delta(t+2) = ( 1+sqrt(1+4*delta(t+1)^2) )/2;

	%[t,z1']
	if sum(abs(z1-z0)) <= 1e-4
		break;
	end
end
z = z1;
clear z0;
clear z1;
clear delta;
clear beta;


function S = SimplexPr(X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% SimplexPr
% Written by Wei Liu (wliu@ee.columbia.edu)
% X(CXN): input data matrix, C: dimension, N: # samples
% S: the projected matrix of X onto C-dimensional simplex  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[C,N] = size(X);
[T1,T2] = sort(X,1,'descend');
clear T2;
S = X;

for i = 1:N
	kk = 0;
	t = T1(:,i);
	for j = 1:C
		tep = t(j)-(sum(t(1:j))-1)/j;
		if tep <= 0
			kk = j-1;
			break;
		end
	end

	if kk == 0
		kk = C;
	end
	theta = (sum(t(1:kk))-1)/kk;
	S(:,i) = max(X(:,i)-theta,0);
	clear t;
end
clear T1;

