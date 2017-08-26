% estimate running time

%% env
close all;
clear;
clc;
%addpath(genpath('./dep'));
%addpath(genpath('./baselines'));
%addpath('./lib/flann');
%run('./lib/vlfeat-0.9.20/toolbox/vl_setup.m');
warning off all;
addpath(genpath('./baselines'));
addpath(genpath('./mmlp'));
addpath('./flann-linux');
%addpath('./flann-win');
% parpool(8);

%%
dataset = 'number';
save_path = 'result/run_time';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, dataset);
if ~exist(record_path, 'dir')
    mkdir(record_path);
end

%%
para.iter = 20;
para.type = 'equal';
para.pca_preserve = 50;
para.p = [10];% label number of each class
para.s = 3; % anchor
para.cn = 10;
para.num_anchor = 1000;
para.beta = 1;
para.knn = 3;

%%
[fea, gnd] = make_classification(60000, 50, 10);
X_train = fea';
Y_train = gnd;

%%
num_samples = [3750, 7500, 15000, 30000, 60000];
samples = cell(numel(num_samples), 2);
labels = cell(numel(num_samples), 1);
for i = 1 : numel(num_samples)
    sample_ind = randsample(1:60000, num_samples(i));
    
    X_tmp = X_train(:, sample_ind);
    Y_tmp = Y_train(sample_ind);
    
    [U, M] = pca(X_tmp, 50);
    X_tmp = U'*bsxfun(@minus, X_tmp, M);
    clear U M;
    
    samples{i,1} = X_tmp;
    samples{i,2} = Y_tmp;
    
    l_tmp = generate_label(Y_tmp, para);
    labels{i} = l_tmp;
end

%%
save(fullfile(record_path, 'samples.mat'), 'samples', 'labels', 'num_samples');

%%
load(fullfile(record_path, 'samples.mat'));

%%
ags = cell(numel(num_samples), 5);
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
  
    [~, anchor, kmeans_time] = k_means(X_tmp, para.num_anchor);
    [B, rL, ag_time] = flann_AnchorGraph(X_tmp, anchor, para.s, 1, para.cn);
    
    ags{i,1} = B;
    ags{i,2} = rL;
    ags{i,3} = ag_time;
    ags{i,4} = kmeans_time;
    ags{i,5} = anchor;
    
    fprintf('AG: num=%d, kmeans_time=%f, ag_time=%f\n', ...
            num_samples(i), kmeans_time, ag_time);
end

%%
save(fullfile(record_path, 'ags.mat'), 'ags');

%%
load(fullfile(record_path, 'ags.mat'));

%%
eags = cell(numel(num_samples), 5);
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    
    anchor = ags{i,5};
    kmeans_time = ags{i,4};

    tic;
    [Z] = FLAE(anchor', X_train', para.knn, para.beta(i));    
    W=Z'*Z; % Normalized graph Laplacian
    Dt=diag(sum(W).^(-1/2));
    S=Dt*W*Dt;
    rLz=eye(para.num_anchor,para.num_anchor)-S; 
    eag_time = toc;
    
    save(eag_data, 'Z', 'rLz', 'eag_time', 'kmeans_time', 'anchor');
    
    eags{i,1} = Z;
    eags{i,2} = rLz;
    eags{i,3} = eag_time;
    eags{i,4} = kmeans_time;
    eags{i,5} = anchor;
    
    fprintf('EAG: num=%d, kmeans_time=%f, eag_time=%f\n', ...
            num_samples(i), kmeans_time, eag_time);
end

%%
save(fullfile(record_path, 'eags.mat'), 'eags');

%%
load(fullfile(record_path, 'eags.mat'));

%%
lgs = cell(numel(num_samples), 12);
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    n = size(X_tmp, 2);
    
    %tic;S = constructS(X_tmp, 10);s_time=toc;
    [S, s_time] = knn_graph_max(X_tmp, 11);
    
    s = 1e-5/10;

    tic;
    [ii, jj, ss] = find(S);
    [mm, nn] = size(S);
    s_mean = mean(ss);
    para_t = - full(s_mean) ./ log(s);
    W = sparse(ii, jj, exp(-ss ./ para_t), mm, nn);
    W(isnan(W)) = 0; W(isinf(W)) = 0;
    w_time=toc;

    tic;
    D = spdiags(sum(W, 2), 0, nn, nn);
    L = D - W;
    D(isnan(D)) = 0; D(isinf(D)) = 0;
    L(isnan(L)) = 0; L(isinf(L)) = 0;
    l_time=toc;

    tic;
    alpha = 0.99; % default value
    nD = spdiags(sum(W, 2).^(-0.5), 0, nn, nn);
    nD(isnan(nD)) = 0; nD(isinf(nD)) = 0;
    nL = speye(size(W)) - alpha .* (nD * W * nD);
    nL(isnan(nL)) = 0; nL(isinf(nL)) = 0;
    nl_time=toc;
    
    %[E2, mmlp_gr_time] = knn_graph2(X_tmp, 11);
    [E2, mmlp_gr_time] = knn_graph_min(X_tmp, 11);
    
    tic;
    [idx1, idx2] = find(W~=0);
    edges = [idx1 idx2 W(idx1+(idx2-1).*n)];
    mtc_gr_time = toc;
    
    lgs{i,1} = S;
    lgs{i,2} = s_time;
    lgs{i,3} = W;
    lgs{i,4} = w_time;
    lgs{i,5} = L;
    lgs{i,6} = l_time;
    lgs{i,7} = nL;
    lgs{i,8} = nl_time;
    lgs{i,9} = E2;
    lgs{i,10} = mmlp_gr_time;
    lgs{i,11} = edges;
    lgs{i,12} = mtc_gr_time;
    
    fprintf('LG: num=%d, s_time=%f, w_time=%f, l_time=%f, nl_time=%f, mmlp_gr_time=%f, mtc_gr_time=%f\n', ...
            num_samples(i), s_time, w_time, l_time, nl_time, mmlp_gr_time, mtc_gr_time);
end

%%
save(fullfile(record_path, 'lgs.mat'), 'lgs');

%%
load(fullfile(record_path, 'lgs.mat'));

%%
class = unique(samples{5,2});
n_class = numel(class);
    
%%
FME_time = zeros(numel(num_samples), 20);
p.ul = 1e9;
p.uu = 0;
p.mu = 1e-9;
p.gamma = 1e-9;
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    L_tmp = lgs{i,5};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples(i), n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == class(cc));
            Y(label_ind(cc_ind),cc) = 1;
        end
        Y = sparse(Y);
        [W, b, F_train] = FME_semi_v2(X_tmp, L_tmp, Y, p);
        FME_time(i, t) = toc;
        fprintf('FME: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, FME_time(i, t));
    end
end

%%
save(fullfile(record_path, 'fme.mat'), 'FME_time');

%%
load(fullfile(record_path, 'fme.mat'));

%%
FME_time_ver = zeros(numel(num_samples), 20);
p.ul = 1e9;
p.uu = 0;
p.mu = 1e-9;
p.lamda = 1e-9;
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    L_tmp = lgs{i,5};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples(i), n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == class(cc));
            Y(label_ind(cc_ind),cc) = 1;
        end
        Y = sparse(Y);
        [W, b, F_train] = FME_semi(X_tmp, L_tmp, Y, p);
        FME_time_ver(i, t) = toc;
        fprintf('FME: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, FME_time_ver(i, t));
    end
end

%%
save(fullfile(record_path, 'fme_ver.mat'), 'FME_time_ver');

%%
GFHF_time = zeros(numel(num_samples), 20);
for i = 1 : numel(num_samples)
    Y_tmp = samples{i,2};
    L_tmp = lgs{i,5};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        % construct Y
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples(i), n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == cc);
            Y(label_ind(cc_ind), cc) = 1;
        end
        Y = sparse(Y);
        % compute F
        label_ind = find(l_tmp(:,t));
        unlabel_ind = find(~l_tmp(:,t));
        F = - L_tmp(unlabel_ind, unlabel_ind) \ ...
            (L_tmp(unlabel_ind, label_ind) * Y(label_ind, :)); 
        % normalization
        q = sum(Y(label_ind,:),1) + 1;
        F = F .* repmat(q ./ sum(F, 1), numel(unlabel_ind), 1);      
        GFHF_time(i, t) = toc;
        fprintf('GFHF: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, GFHF_time(i, t));
    end
end

%%
save(fullfile(record_path, 'gfhf.mat'), 'GFHF_time');

%%
load(fullfile(record_path, 'gfhf.mat'));

%%
LGC_time = zeros(numel(num_samples), 20);
for i = 1 : numel(num_samples)
    Y_tmp = samples{i,2};
    nL_tmp = lgs{i,7};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        % construct Y
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples(i), n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == cc);
            Y(label_ind(cc_ind), cc) = 1;
        end
        Y = sparse(Y);
        % compute F
        F = nL_tmp \ Y; clear Y;
        LGC_time(i, t) = toc;
        fprintf('LGC: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, LGC_time(i, t));
    end
end

%%
save(fullfile(record_path, 'lgc.mat'), 'LGC_time');

%%
load(fullfile(record_path, 'lgc.mat'));

%%
AGR_time = zeros(numel(num_samples), 20);
for i = 1 : numel(num_samples)
    Y_tmp = samples{i,2};
    B_tmp = ags{i,1};
    rL_tmp = ags{i,2};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        label_ind = find(l_tmp(:,t));
        tic;
        [~, ~, e] = AnchorGraphReg(B_tmp, rL_tmp, Y_tmp', label_ind, 0.01);
        AGR_time(i, t) = toc;
        fprintf('AGR: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, AGR_time(i, t));
    end
end

%%
save(fullfile(record_path, 'agr.mat'), 'AGR_time');

%%
load(fullfile(record_path, 'agr.mat'));

%%
MMLP_time = zeros(numel(num_samples), 20);
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    E_tmp = lgs{i,9};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        label_ind = find(l_tmp(:,t));
        [~, e, ~, ~, MMLP_time(i, t)] = mmlp(E_tmp, X_tmp, Y_tmp, label_ind);
        fprintf('MMLP: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, MMLP_time(i, t));
    end
end

%%
save(fullfile(record_path, 'mmlp.mat'), 'MMLP_time');

%%
load(fullfile(record_path, 'mmlp.mat'));

%%
MTC_time = zeros(numel(num_samples), 20);
for i = 1 : numel(num_samples)
    Y_tmp = samples{i,2};
    e_tmp = lgs{i,11};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        % construct Y    
        Y = zeros(num_samples(i), 1)-1;
        Y(label_ind) = Y_tmp(label_ind) - 1;
        % compute F
        F = mtc_matlab(full(e_tmp), num_samples(i), Y, n_class, 0, 1);
        F = F + 1;
        MTC_time(i, t) = toc;
        fprintf('MTC: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, MTC_time(i, t));
    end
end

%%
save(fullfile(record_path, 'mtc.mat'), 'MTC_time');

%%
load(fullfile(record_path, 'mtc.mat'));

%%
LAPRLS_time = zeros(numel(num_samples), 20);
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    L_tmp = lgs{i,5};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        
        label_ind = find(l_tmp(:,t));
        gammaA = 1e-3; gammaI = 1e-3;        
        
        nFea = size(X_tmp, 1);
        % construct the labeled matrix
        feaLabel = X_tmp(:, label_ind);
        gndLabel = Y_tmp(label_ind);
        classLabel = unique(gndLabel);
        nClass = numel(classLabel);
        nLabel = numel(gndLabel);
        YLabel = zeros(nLabel, nClass);
        for cc = 1 : nClass
            YLabel(gndLabel == classLabel(cc), cc) = 1;
        end
        % compute W
        Xl = bsxfun(@minus, feaLabel, mean(feaLabel,2));
        W = (Xl * Xl' + gammaA * nLabel .* eye(nFea) + ...
            gammaI * nLabel .* (X_tmp * L_tmp * X_tmp')) \ (Xl * YLabel);
        b = 1/nLabel*(sum(YLabel,1)' - W'*(feaLabel*ones(nLabel,1)));
        
        LAPRLS_time(i, t) = toc;
        fprintf('MTC: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, LAPRLS_time(i, t));
    end
end

%%
save(fullfile(record_path, 'laprls.mat'), 'LAPRLS_time');

%%
load(fullfile(record_path, 'laprls.mat'));

%%
fFME_time = zeros(numel(num_samples), 20);
p.ul = 1e9;
p.uu = 0;
p.mu = 1e-9;
p.gamma = 1e-9;
for i = 1 : numel(num_samples)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    B_tmp = ags{i,1};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples(i), n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == class(cc));
            Y(label_ind(cc_ind),cc) = 1;
        end
        Y = sparse(Y);
        [W, b, F_train] = fastFME_semi(X_tmp, B_tmp, Y, p);
        fFME_time(i, t) = toc;
        fprintf('fFME: num=%d, t=%d, time=%f\n', ...
            num_samples(i), t, fFME_time(i, t));
    end
end

%%
save(fullfile(record_path, 'ffme.mat'), 'fFME_time');

%%
load(fullfile(record_path, 'ffme.mat'));
