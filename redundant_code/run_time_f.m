% estimate running time

%% env
close all;
clear;
clc;
warning off all;
addpath(genpath('./baselines'));
addpath(genpath('./mmlp'));
addpath('./flann-linux');
addpath('./framework');
addpath('./fFME');
%addpath('./flann-win');
% parpool(8);

%%
dataset = 'feature';
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
para.p = [10];% label number of each class
para.s = 3; % anchor
para.cn = 10;
para.num_anchor = 1000;
para.beta = 1;
para.knn = 3;

%%
num_samples = 10000;
num_features = [512, 1024, 2048, 4096, 8192];
samples = cell(numel(num_features), 2);
labels = cell(numel(num_features), 1);
for i = 1 : numel(num_features)
    [fea, gnd] = make_classification(num_samples, num_features(i), 10);
    
    samples{i,1} = fea';
    samples{i,2} = gnd;
    
    l_tmp = generate_label(gnd, para);
    labels{i} = l_tmp;
end

%%
save(fullfile(record_path, 'samples.mat'), 'samples', 'labels', 'num_features');

%%
load(fullfile(record_path, 'samples.mat'));

%%
ags = cell(numel(num_features), 5);
for i = 1 : numel(num_features)
    X_tmp = samples{i,1};
  
    [~, anchor, kmeans_time] = k_means(X_tmp, para.num_anchor);
    [B, rL, ag_time] = flann_AnchorGraph(X_tmp, anchor, para.s, 1, para.cn);
    
    ags{i,1} = B;
    ags{i,2} = rL;
    ags{i,3} = ag_time;
    ags{i,4} = kmeans_time;
    ags{i,5} = anchor;
    
    fprintf('AG: num_feature=%d, kmeans_time=%f, ag_time=%f\n', ...
            num_features(i), kmeans_time, ag_time);
end

%%
save(fullfile(record_path, 'ags.mat'), 'ags');

%%
load(fullfile(record_path, 'ags.mat'));

%%
eags = cell(numel(num_features), 5);
for i = 1 : numel(num_features)
    X_tmp = samples{i,1};
    
    anchor = ags{i,5};
    kmeans_time = ags{i,4};

    tic;
    [Z] = FLAE(anchor', X_tmp', para.knn, para.beta);    
    W=Z'*Z; % Normalized graph Laplacian
    Dt=diag(sum(W).^(-1/2));
    S=Dt*W*Dt;
    rLz=eye(para.num_anchor,para.num_anchor)-S; 
    eag_time = toc;
    
    eags{i,1} = Z;
    eags{i,2} = rLz;
    eags{i,3} = eag_time;
    eags{i,4} = kmeans_time;
    eags{i,5} = anchor;
    
    fprintf('EAG: num_feature=%d, kmeans_time=%f, eag_time=%f\n', ...
            num_features(i), kmeans_time, eag_time);
end

%%
save(fullfile(record_path, 'eags.mat'), 'eags');

%%
load(fullfile(record_path, 'eags.mat'));

%%
lgs = cell(numel(num_features), 12);
for i = 1 : numel(num_features)
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
    
    fprintf('LG: num_feature=%d, s_time=%f, w_time=%f, l_time=%f, nl_time=%f, mmlp_gr_time=%f, mtc_gr_time=%f\n', ...
            num_features(i), s_time, w_time, l_time, nl_time, mmlp_gr_time, mtc_gr_time);
end

%%
save(fullfile(record_path, 'lgs.mat'), 'lgs');

%%
load(fullfile(record_path, 'lgs.mat'));

%%
class = unique(samples{1,2});
n_class = numel(class);

%%
AGR_time = zeros(numel(num_features), 20);
for i = 1 : numel(num_features)
    Y_tmp = samples{i,2};
    B_tmp = ags{i,1};
    rL_tmp = ags{i,2};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        label_ind = find(l_tmp(:,t));
        tic;
        [~, ~, e] = AnchorGraphReg(B_tmp, rL_tmp, Y_tmp', label_ind, 0.01);
        AGR_time(i, t) = toc;
        fprintf('AGR: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, AGR_time(i, t));
    end
end
%
save(fullfile(record_path, 'agr.mat'), 'AGR_time');

%%
load(fullfile(record_path, 'agr.mat'));

%%
EAGR_time = zeros(numel(num_features), 20);
for i = 1 : numel(num_features)
    Y_tmp = samples{i,2};
    Z_tmp = eags{i,1};
    rLz_tmp = eags{i,2};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        label_ind = find(l_tmp(:,t));
        tic;
        [acc, F] = EAGReg(Z_tmp, rLz_tmp, Y_tmp', label_ind, 1);
        EAGR_time(i, t) = toc;
        fprintf('EAGR: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, EAGR_time(i, t));
    end
end
save(fullfile(record_path, 'eagr.mat'), 'EAGR_time');

%%
MMLP_time = zeros(numel(num_features), 20);
for i = 1 : numel(num_features)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    E_tmp = lgs{i,9};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        label_ind = find(l_tmp(:,t));
        [~, e, ~, ~, MMLP_time(i, t)] = mmlp(E_tmp, X_tmp, Y_tmp, label_ind);
        fprintf('MMLP: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, MMLP_time(i, t));
    end
end

%
save(fullfile(record_path, 'mmlp.mat'), 'MMLP_time');

%%
load(fullfile(record_path, 'mmlp.mat'));

%%
MTC_time = zeros(numel(num_features), 20);
for i = 1 : numel(num_features)
    Y_tmp = samples{i,2};
    e_tmp = lgs{i,11};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        % construct Y    
        Y = zeros(num_samples, 1)-1;
        Y(label_ind) = Y_tmp(label_ind) - 1;
        % compute F
        F = mtc_matlab(full(e_tmp), num_samples, Y, n_class, 0, 1);
        F = F + 1;
        MTC_time(i, t) = toc;
        fprintf('MTC: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, MTC_time(i, t));
    end
end

%
save(fullfile(record_path, 'mtc.mat'), 'MTC_time');

%%
load(fullfile(record_path, 'mtc.mat'));

%%
LAPRLS_time = zeros(numel(num_features), 20);
for i = 1 : numel(num_features)
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
        fprintf('laprls: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, LAPRLS_time(i, t));
    end
end

%
save(fullfile(record_path, 'laprls.mat'), 'LAPRLS_time');

%%
load(fullfile(record_path, 'laprls.mat'));

%%
fFME_time = zeros(numel(num_features), 20);
p.ul = 1e9;
p.uu = 0;
p.mu = 1e-9;
p.gamma = 1e-9;
for i = 1 : numel(num_features)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    B_tmp = ags{i,1};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples, n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == class(cc));
            Y(label_ind(cc_ind),cc) = 1;
        end
        Y = sparse(Y);
        [W, b, F_train] = fastFME_semi(X_tmp, B_tmp, Y, p);
        fFME_time(i, t) = toc;
        fprintf('fFME: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, fFME_time(i, t));
    end
end

%
save(fullfile(record_path, 'ffme.mat'), 'fFME_time');

%%
load(fullfile(record_path, 'ffme.mat'));

%%
efFME_time = zeros(numel(num_features), 20);
p.ul = 1e9;
p.uu = 0;
p.mu = 1e-9;
p.gamma = 1e-9;
for i = 1 : numel(num_features)
    X_tmp = samples{i,1};
    Y_tmp = samples{i,2};
    Z_tmp = eags{i,1};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples, n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == class(cc));
            Y(label_ind(cc_ind),cc) = 1;
        end
        Y = sparse(Y);
        [W, b, F_train] = fastFME_semi(X_tmp, Z_tmp, Y, p);
        efFME_time(i, t) = toc;
        fprintf('efFME: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, efFME_time(i, t));
    end
end
save(fullfile(record_path, 'effme.mat'), 'efFME_time');

%%
aFME_time = zeros(numel(num_features), 20);
p.ul = 1e9;
p.uu = 0;
p.mu = 1e-9;
p.gamma = 1e-9;
for i = 1 : numel(num_features)
    anchor = eags{i,5};
    Y_tmp = samples{i,2};
    Z_tmp = eags{i,1};
    rLz_tmp = eags{i,2};
    l_tmp = labels{i}{1};
    for t = 1 : 20
        tic;
        label_ind = find(l_tmp(:,t));
        Y = zeros(num_samples, n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_tmp(label_ind) == class(cc));
            Y(label_ind(cc_ind),cc) = 1;
        end
        Y = sparse(Y);
        [W, b, F_train] = aFME_semi(anchor, Z_tmp, rLz_tmp, Y, p);
        aFME_time(i, t) = toc;
        fprintf('aFME: num_feature=%d, t=%d, time=%f\n', ...
            num_features(i), t, aFME_time(i, t));
    end
end
save(fullfile(record_path, 'afme.mat'), 'aFME_time');

%%
load(fullfile(record_path, 'afme.mat'));
load(fullfile(record_path, 'effme.mat'));
load(fullfile(record_path, 'laprls.mat'));

aFME_time = mean(aFME_time, 2)'
efFME_time = mean(efFME_time, 2)'
LAPRLS_time = mean(LAPRLS_time, 2)'

fileID = fopen(fullfile(record_path, 'feature_run_time.txt'),'w');
fprintf(fileID, '\\#Features & %d & %d & %d & %d & %d \\\\ \n', num_features);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'fFME', efFME_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'LAPRLS', LAPRLS_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'aFME', aFME_time);
fclose(fileID);