
addpath(genpath('./baselines'));
addpath(genpath('./mmlp'));
addpath(genpath('./framework'));
addpath(genpath('./fFME'));
addpath('./flann-linux');

%% para
save_path = 'result/test-model';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, 'record-corners-2');
if ~exist(record_path, 'dir')
    mkdir(record_path);
end

para.iter = 20;
para.type = 'equal';
para.K = 10;
para.p = 2;% label number of each class
para.s = 3; % anchor
para.cn = 10;
para.knn = 3;
para.beta = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
para.num_anchor = 100;
save(fullfile(record_path, 'para.mat'), 'para');

%% random data
%[fea, gnd] = my_two_moon(1000);
%data = halfkernel(2000,-20,20,35,7);
data = corners(2000);
fea=data(:,1:2); gnd=data(:,3);
fea=fea'; gnd=gnd+1;
gscatter(fea(1,:)', fea(2,:)', gnd);
print(gcf,'-dpng',fullfile(record_path, 'data.png'));

%% load data
pca_data = fullfile(record_path, 'pca.mat');
if ~exist(pca_data, 'file')
    % default split
    split = choose_each_class(gnd, 0.5, 1);
    % preprocess
    X_train = fea(:, split); Y_train = gnd(split);
    X_test = fea(:, ~split); Y_test = gnd(~split);
    % save
    save(pca_data, 'X_train', 'Y_train', 'X_test', 'Y_test');
else
    load(pca_data);
end

%% generate label
label_data = fullfile(record_path, 'label.mat');
if ~exist(label_data, 'file')
    label = generate_label(Y_train, para);
    save(label_data, 'label');
else
    load(label_data);
end

%% compute anchor graph
ag_data = fullfile(record_path, 'ag.mat');
if ~exist(ag_data, 'file')
    [~, anchor, kmeans_time] = k_means(X_train, para.num_anchor);
    [B, rL, ag_time] = flann_AnchorGraph(X_train, anchor, para.s, 1, para.cn);
    save(ag_data, 'B', 'rL', 'ag_time', 'kmeans_time', 'anchor');
else
    load(ag_data);
end

%% run AGR
best_gamma = [];
agr_data_para_best = fullfile(record_path, 'result_AGR_para_best.mat');
if ~exist(agr_data_para_best, 'file')
    result_AGR_para_best = run_AGR_para(Y_train, B, rL, label, best_gamma);
    save(agr_data_para_best, 'result_AGR_para_best');
else
    load(agr_data_para_best);
end
celldisp(result_AGR_para_best);

%% run fast FME
best_mu = [0]; 
best_gamma = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best.mat');
if ~exist(ffme_data_1_1e9_para_best, 'file')
    result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, label, ...
        1e-6, best_mu, best_gamma, true);
    save(ffme_data_1_1e9_para_best, 'result_fastFME1_1e9_para_best');
else
    load(ffme_data_1_1e9_para_best);
end
celldisp(result_fastFME1_1e9_para_best);

%% run fast FME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best2.mat');
if ~exist(ffme_data_1_1e9_para_best, 'file')
    result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, label, ...
        1e9, mu, gamma, true);
    save(ffme_data_1_1e9_para_best, 'result_fastFME1_1e9_para_best');
else
    load(ffme_data_1_1e9_para_best);
end
celldisp(result_fastFME1_1e9_para_best);

%% compute efficient anchor graph
eag_data = fullfile(record_path, 'eag.mat');
if ~exist(eag_data, 'file')
    %[~, anchor, kmeans_time] = k_means(X_train, para.num_anchor);
    Z = cell(numel(para.beta), 1);
    rLz = cell(numel(para.beta), 1);
    eag_time = zeros(numel(para.beta), 1);
    for i = 1:numel(para.beta)
        tic;[Z{i}] = FLAE(anchor', X_train', para.knn, para.beta(i));
        % Normalized graph Laplacian
        W=Z{i}'*Z{i};
        Dt=diag(sum(W).^(-1/2));
        S=Dt*W*Dt;
        rLz{i}=eye(para.num_anchor,para.num_anchor)-S; eag_time(i) = toc;
    end
    save(eag_data, 'Z', 'rLz', 'eag_time', 'kmeans_time', 'anchor');
else
    load(eag_data);
end

%% run EAGR
best_gamma = [];
eagr_data_para_best = fullfile(record_path, 'result_EAGR_para_best.mat');
if ~exist(eagr_data_para_best, 'file')
    result_EAGR_para_best = run_EAGR_para(Y_train, Z, rLz, label, best_gamma, para);
    save(eagr_data_para_best, 'result_EAGR_para_best');
else
    load(eagr_data_para_best);
end
celldisp(result_EAGR_para_best);

%% run aFME
mu = [0];
gamma = [1];
best_beta = result_EAGR_para_best{1}.best_id(1)
afme_data_1e9_para_best = fullfile(record_path, 'result_aFME_1e9_para_best1.mat');
if ~exist(afme_data_1e9_para_best, 'file')
    result_aFME_1e9_para_best = run_aFME_semi_para(X_train, Y_train, X_test, Y_test, anchor, ...
        Z{best_beta}, rLz{best_beta}, label, 1e9, mu, gamma, true);
    save(afme_data_1e9_para_best, 'result_aFME_1e9_para_best');
else
    load(afme_data_1e9_para_best);
end
celldisp(result_aFME_1e9_para_best);

%% run aFME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
best_beta = result_EAGR_para_best{1}.best_id(1)
afme_data_1e9_para_best = fullfile(record_path, 'result_aFME_1e9_para_best.mat');
if ~exist(afme_data_1e9_para_best, 'file')
    result_aFME_1e9_para_best = run_aFME_semi_para(X_train, Y_train, X_test, Y_test, anchor, ...
        Z{best_beta}, rLz{best_beta}, label, 1e9, mu, gamma, true);
    save(afme_data_1e9_para_best, 'result_aFME_1e9_para_best');
else
    load(afme_data_1e9_para_best);
end
celldisp(result_aFME_1e9_para_best);

%% E_min
emin_data = fullfile(record_path, 'E_min.mat');
if ~exist(emin_data, 'file')
    [E_min, mmlp_gr_time_min] = knn_graph_min(X_train, para.K+1);
    save(emin_data, 'E_min', 'mmlp_gr_time_min');
else
    load(emin_data);
end

% E_max
emax_data = fullfile(record_path, 'E_max.mat');
if ~exist(emax_data, 'file')
    [E_max, mmlp_gr_time_max] = knn_graph_max(X_train, para.K+1);
    save(emax_data, 'E_max', 'mmlp_gr_time_max');
else
    load(emax_data);
end

%% run MMLP
mmlp_data_para = fullfile(record_path, 'result_MMLP_min_para.mat');
if ~exist(mmlp_data_para, 'file')
    result_MMLP_min_para = run_MMLP_para(X_train, Y_train, E_min, label);
    save(mmlp_data_para, 'result_MMLP_min_para');
else
    load(mmlp_data_para);
end

mmlp_data_para = fullfile(record_path, 'result_MMLP_max_para.mat');
if ~exist(mmlp_data_para, 'file')
    result_MMLP_max_para = run_MMLP_para(X_train, Y_train, E_max, label);
    save(mmlp_data_para, 'result_MMLP_max_para');
else
    load(mmlp_data_para);
end

%%
if result_MMLP_min_para{1}.best_train_accuracy(1) >= result_MMLP_max_para{1}.best_train_accuracy(1)
    result_MMLP_para = result_MMLP_min_para;
else
    result_MMLP_para = result_MMLP_max_para;
end
celldisp(result_MMLP_para)

%% run MTC
best_s = [];
mtc_data_para = fullfile(record_path, 'result_MTC_para.mat');
if ~exist(mtc_data_para, 'file')
    result_MTC_para = run_MTC_para(Y_train, E_max, label, best_s);
    save(mtc_data_para, 'result_MTC_para');
else
    load(mtc_data_para);
end

%% run NN
nn_data_para = fullfile(record_path, 'result_NN_para.mat');
if ~exist(nn_data_para, 'file')
    result_NN_para = run_NN_para(X_train, Y_train, X_test, Y_test, label);
    save(nn_data_para, 'result_NN_para');
else
    load(nn_data_para);
end
celldisp(result_NN_para);

%% run LapRLS/L
best_s = []; best_gammaA = []; best_gammaI = [];
laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');
if ~exist(laprls_data2_para_best, 'file')
    result_LapRLS2_para_best = run_LapRLS2_para(X_train, Y_train, X_test, Y_test, E_max, label, ...
        best_s, best_gammaA, best_gammaI);
    save(laprls_data2_para_best, 'result_LapRLS2_para_best');
else
    load(laprls_data2_para_best);
end
celldisp(result_LapRLS2_para_best);

%% run RS
best_gammaA = [];
result_RS = run_RidgeRegression_para(X_train, Y_train, X_test, Y_test, label, best_gammaA);
celldisp(result_RS);

best_gammaA = [];
result_RS = run_RidgeRegression_para(X_train, Y_train, X_test, Y_test, {true(1000,2)}, best_gammaA);
celldisp(result_RS);

%% display
celldisp(result_AGR_para_best);
celldisp(result_EAGR_para_best);
celldisp(result_MMLP_para);
celldisp(result_MTC_para);
celldisp(result_NN_para);
celldisp(result_LapRLS2_para_best);
celldisp(result_fastFME1_1e9_para_best);
celldisp(result_aFME_1e9_para_best);