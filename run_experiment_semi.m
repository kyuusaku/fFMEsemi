function run_experiment_semi(dataset,o,varargin)
% code for running the scalable semi supervised learning algorithms

%% env
close all;
clc;
warning off all;
% add necessary paths
addpath(genpath('./baselines'));
addpath(genpath('./mmlp'));
addpath(genpath('./framework'));
addpath(genpath('./fFME'));
% parse inputs
p = parse_inputs();
parse(p,dataset,o,varargin{:});
% 
addpath(['./flann-' p.Results.system]);
if p.Results.parfor
    parpool(p.Results.parforNumber);
end

%% para
data_name = get_data_name(p.Results.dataset);
save_path = ['result/' p.Results.dataset '/semi'];
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, ['record-' num2str(p.Results.o)]);
if ~exist(record_path, 'dir')
    mkdir(record_path);
end

para.dataset = data_name;
para.dataPath = fullfile(pwd, 'data');
para.iter = 20;
para.type = 'equal';
para.pca_preserve = 50;
para.p = [o];% number of label per class
para.s = 3; % anchor
para.cn = 10;
para.num_anchor = 1000;
para.knn = 3;
para.beta = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
para.num_anchors = get_num_anchors(p.Results.dataset);
para.K = 10;
save(fullfile(record_path, 'para.mat'), 'para');

%% load data
pca_data = fullfile(save_path, 'pca.mat');
if ~exist(pca_data, 'file')
    [X_train, Y_train, X_test, Y_test] = load_dataset(p.Results.dataset,para);
    % check gnd
    tmp_class = unique(Y_train)';
    if sum(tmp_class == 1:numel(tmp_class)) == numel(tmp_class)
        fprintf('input label is ok, total class: %d\n', numel(tmp_class));
    else
        error('input label must be a number sequence from 1 to c (c is the number of classes)');
    end
    tmp_class = unique(Y_test)';
    if sum(tmp_class == 1:numel(tmp_class)) == numel(tmp_class)
        fprintf('input label is ok, total class: %d\n', numel(tmp_class));
    else
        error('input label must be a number sequence from 1 to c (c is the number of classes)');
    end
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
ag_data = fullfile(save_path, 'ag.mat');
if ~exist(ag_data, 'file')
    [~, anchor, kmeans_time] = k_means(X_train, para.num_anchor);
    [B, rL, ag_time] = flann_AnchorGraph(X_train, anchor, para.s, 1, para.cn);
    save(ag_data, 'B', 'rL', 'ag_time', 'kmeans_time', 'anchor');
else
    load(ag_data);
end

%% compute efficient anchor graph
eag_data = fullfile(save_path, 'eag.mat');
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

%% compute anchor hierarchy
% hag_data = fullfile(save_path, 'hag.mat');
% if ~exist(hag_data, 'file')
%     [anchors, anchors_time] = hanchors(X_train, para.num_anchors);
%     [ZH, Z01, zh_time, rLh, rLh_time] = hag(X_train, anchors, 3);
%     save(hag_data, 'ZH', 'Z01', 'rLh', 'zh_time', 'rLh_time', 'anchors', 'anchors_time');
% else
%     load(hag_data);
% end

%% run fast FME
% best_mu = []; best_gamma = [];
% ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best.mat');
% if ~exist(ffme_data_1_1e9_para_best, 'file')
%     result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, label, ...
%         1e9, best_mu, best_gamma);
%     save(ffme_data_1_1e9_para_best, 'result_fastFME1_1e9_para_best');
% else
%     load(ffme_data_1_1e9_para_best);
% end
% celldisp(result_fastFME1_1e9_para_best);

%% run fast FME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best2.mat');
if ~exist(ffme_data_1_1e9_para_best, 'file')
    result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, label, ...
        1e9, mu, gamma);
    save(ffme_data_1_1e9_para_best, 'result_fastFME1_1e9_para_best');
else
    load(ffme_data_1_1e9_para_best);
end
celldisp(result_fastFME1_1e9_para_best);

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

%% run HAGR
% best_gamma = [];
% hagr_data_para_best = fullfile(record_path, 'result_HAGR_para_best.mat');
% if ~exist(hagr_data_para_best, 'file')
%     result_HAGR_para_best = run_HAGR_para(Y_train, ZH, rLh, label, best_gamma);
%     save(hagr_data_para_best, 'result_HAGR_para_best');
% else
%     load(hagr_data_para_best);
% end
% celldisp(result_HAGR_para_best);

%% run eFME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
best_beta = result_EAGR_para_best{1}.best_id(1)
efme_data_1e9_para_best = fullfile(record_path, 'result_eFME_1e9_para_best.mat');
if ~exist(efme_data_1e9_para_best, 'file')
    result_eFME_1e9_para_best = run_eFME_semi_para(X_train, Y_train, X_test, Y_test, ...
        Z{best_beta}, rLz{best_beta}, label, 1e9, mu, gamma);
    save(efme_data_1e9_para_best, 'result_eFME_1e9_para_best');
else
    load(efme_data_1e9_para_best);
end
celldisp(result_eFME_1e9_para_best);

%% run efFME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
best_beta = result_EAGR_para_best{1}.best_id(1)
effme_data_1e9_para_best = fullfile(record_path, 'result_efFME_1e9_para_best.mat');
if ~exist(effme_data_1e9_para_best, 'file')
    result_efFME_1e9_para_best = run_efFME_semi_para(X_train, Y_train, X_test, Y_test, ...
        Z{best_beta}, label, 1e9, mu, gamma);
    save(effme_data_1e9_para_best, 'result_efFME_1e9_para_best');
else
    load(effme_data_1e9_para_best);
end
celldisp(result_efFME_1e9_para_best);

%% run aFME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
best_beta = result_EAGR_para_best{1}.best_id(1)
afme_data_1e9_para_best = fullfile(record_path, 'result_aFME_1e9_para_best.mat');
if ~exist(afme_data_1e9_para_best, 'file')
    result_aFME_1e9_para_best = run_aFME_semi_para(X_train, Y_train, X_test, Y_test, anchor, ...
        Z{best_beta}, rLz{best_beta}, label, 1e9, mu, gamma);
    save(afme_data_1e9_para_best, 'result_aFME_1e9_para_best');
else
    load(afme_data_1e9_para_best);
end
celldisp(result_aFME_1e9_para_best);

%% run heFME
% mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
% gamma = mu;
% hefme_data_1e9_para_best = fullfile(record_path, 'result_heFME_1e9_para_best.mat');
% if ~exist(hefme_data_1e9_para_best, 'file')
%     result_heFME_1e9_para_best = run_heFME_semi_para(X_train, Y_train, X_test, Y_test, ...
%         ZH, rLh, label, 1e9, mu, gamma);
%     save(hefme_data_1e9_para_best, 'result_heFME_1e9_para_best');
% else
%     load(hefme_data_1e9_para_best);
% end
% celldisp(result_heFME_1e9_para_best);

%% run hfFME
% mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
% gamma = mu;
% hffme_data_1e9_para_best = fullfile(record_path, 'result_hfFME_1e9_para_best.mat');
% if ~exist(hffme_data_1e9_para_best, 'file')
%     result_hfFME_1e9_para_best = run_hfFME_semi_para(X_train, Y_train, X_test, Y_test, ZH, label, ...
%         1e9, mu, gamma);
%     save(hffme_data_1e9_para_best, 'result_hfFME_1e9_para_best');
% else
%     load(hffme_data_1e9_para_best);
% end
% celldisp(result_hfFME_1e9_para_best);

%% E_min
emin_data = fullfile(save_path, 'E_min.mat');
if ~exist(emin_data, 'file')
    [E_min, mmlp_gr_time_min] = knn_graph_min(X_train, para.K+1);
    save(emin_data, 'E_min', 'mmlp_gr_time_min');
else
    load(emin_data);
end

% E_max
emax_data = fullfile(save_path, 'E_max.mat');
if ~exist(emax_data, 'file')
    [E_max, mmlp_gr_time_max] = knn_graph_max(X_train, para.K+1);
    save(emax_data, 'E_max', 'mmlp_gr_time_max');
else
    load(emax_data);
end

%% compute minmax anchor graph
% mmag_data = fullfile(save_path, 'mmag.mat');
% if ~exist(mmag_data, 'file')
%     %[~, anchor, kmeans_time] = k_means(X_train, para.num_anchor);
%     [aIdx, val, find_idx_time] = find_idx(anchor, X_train);
%     [Zm, mmmp_weight_time, ~] = mmmp_weight(E_max, aIdx, para.knn);
%     % Normalized graph Laplacian
%     tic;W=Zm'*Zm;
%     Dt=diag(sum(W).^(-1/2));
%     S=Dt*W*Dt;
%     rLm=eye(para.num_anchor,para.num_anchor)-S; rLm_time = toc;
%     % Save
%     save(mmag_data, 'Zm', 'rLm', 'find_idx_time', 'mmmp_weight_time', ...
%         'rLm_time', 'kmeans_time', 'anchor');
%     clear val W Dt S;
% else
%     load(mmag_data);
% end

%% run mmfFME
% mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
% gamma = mu;
% mmffme_data_1e9_para_best = fullfile(record_path, 'result_mmfFME_1e9_para_best.mat');
% if ~exist(mmffme_data_1e9_para_best, 'file')
%     result_mmfFME_1e9_para_best = run_mmfFME_semi_para(X_train, Y_train, X_test, Y_test, ...
%         Zm, label, 1e9, mu, gamma);
%     save(mmffme_data_1e9_para_best, 'result_mmfFME_1e9_para_best');
% else
%     load(mmffme_data_1e9_para_best);
% end
% celldisp(result_mmfFME_1e9_para_best);

%% run maFME
% mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
% gamma = mu;
% mmafme_data_1e9_para_best = fullfile(record_path, 'result_mmaFME_1e9_para_best.mat');
% if ~exist(mmafme_data_1e9_para_best, 'file')
%     result_mmaFME_1e9_para_best = run_mmaFME_semi_para(X_train, Y_train, X_test, Y_test, anchor, ...
%         Zm, rLm, label, 1e9, mu, gamma);
%     save(mmafme_data_1e9_para_best, 'result_mmaFME_1e9_para_best');
% else
%     load(mmafme_data_1e9_para_best);
% end
% celldisp(result_mmaFME_1e9_para_best);

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



%% ttest
% Unlabel ttest 1=PVM, 2=AGR, 3=MMLP, 4=1NN, 5=LapRLS, 6=fastFME
X_AGR = result_AGR_para_best{1}.accuracy(result_AGR_para_best{1}.best_id, :);
X_EAGR = result_EAGR_para_best{1}.accuracy(...
    result_EAGR_para_best{1}.best_id(1), ...
    result_EAGR_para_best{1}.best_id(2), :);
X_MMLP = result_MMLP_para{1}.accuracy';
X_MTC = result_MTC_para{1}.accuracy(result_MTC_para{1}.best_id, :);
X_NN_u = result_NN_para{1}.accuracy(:,1)';
X_LapRLS_u = result_LapRLS2_para_best{1}.accuracy(...
    result_LapRLS2_para_best{1}.best_train_para_id(1), ...
    result_LapRLS2_para_best{1}.best_train_para_id(2), ...
    result_LapRLS2_para_best{1}.best_train_para_id(3), :, 1);
X_LapRLS_u = squeeze(X_LapRLS_u)';
X_fastFME_u = result_fastFME1_1e9_para_best{1}.accuracy(...
    result_fastFME1_1e9_para_best{1}.best_train_para_id(1), ...
    result_fastFME1_1e9_para_best{1}.best_train_para_id(2), :, 1);
X_fastFME_u = squeeze(X_fastFME_u)';
X_efFME_u = result_efFME_1e9_para_best{1}.accuracy(...
    result_efFME_1e9_para_best{1}.best_train_para_id(1), ...
    result_efFME_1e9_para_best{1}.best_train_para_id(2), :, 1);
X_efFME_u = squeeze(X_efFME_u)';
X_aFME_u = result_aFME_1e9_para_best{1}.accuracy(...
    result_aFME_1e9_para_best{1}.best_train_para_id(1), ...
    result_aFME_1e9_para_best{1}.best_train_para_id(2), :, 1);
X_aFME_u = squeeze(X_aFME_u)';
X_unlabel = {X_AGR; X_EAGR; X_MMLP; X_MTC; X_NN_u; X_LapRLS_u; X_fastFME_u; ...
    X_efFME_u; X_aFME_u};
unlabel_ttest = zeros(numel(X_unlabel), numel(X_unlabel), 2);
for i = 1 : numel(X_unlabel)
    for j = 1 : numel(X_unlabel)
        [unlabel_ttest(i, j, 1), unlabel_ttest(i, j, 2)] = ...
            ttest_my(X_unlabel{i}, X_unlabel{j}, 1, 0.05, 1);
    end
end
% Test ttest 1=1NN, 2=LapRLS, 3=fastFME
X_NN_t = result_NN_para{1}.accuracy(:,2)';
X_LapRLS_t = result_LapRLS2_para_best{1}.accuracy(...
    result_LapRLS2_para_best{1}.best_test_para_id(1), ...
    result_LapRLS2_para_best{1}.best_test_para_id(2), ...
    result_LapRLS2_para_best{1}.best_test_para_id(3), :, 2);
X_LapRLS_t = squeeze(X_LapRLS_t)';
X_fastFME_t = result_fastFME1_1e9_para_best{1}.accuracy(...
    result_fastFME1_1e9_para_best{1}.best_test_para_id(1), ...
    result_fastFME1_1e9_para_best{1}.best_test_para_id(2), :, 2);
X_fastFME_t = squeeze(X_fastFME_t)';
X_efFME_t = result_efFME_1e9_para_best{1}.accuracy(...
    result_efFME_1e9_para_best{1}.best_test_para_id(1), ...
    result_efFME_1e9_para_best{1}.best_test_para_id(2), :, 2);
X_efFME_t = squeeze(X_efFME_t)';
X_aFME_t = result_aFME_1e9_para_best{1}.accuracy(...
    result_aFME_1e9_para_best{1}.best_test_para_id(1), ...
    result_aFME_1e9_para_best{1}.best_test_para_id(2), :, 1);
X_aFME_t = squeeze(X_aFME_t)';
X_test = {X_NN_t; X_LapRLS_t; X_fastFME_t; X_efFME_t; X_aFME_t};
test_ttest = zeros(numel(X_test), numel(X_test), 2);
for i = 1 : numel(X_test)
    for j = 1 : numel(X_test)
        [test_ttest(i, j, 1), test_ttest(i, j, 2)] = ...
            ttest_my(X_test{i}, X_test{j}, 1, 0.05, 1);
    end
end
% save
save(fullfile(record_path, 'ttest.mat'), 'unlabel_ttest', 'test_ttest');

% display
celldisp(result_AGR_para_best);
celldisp(result_EAGR_para_best);
celldisp(result_MMLP_para);
celldisp(result_MTC_para);
celldisp(result_NN_para);
celldisp(result_LapRLS2_para_best);
celldisp(result_fastFME1_1e9_para_best);
celldisp(result_efFME_1e9_para_best);
celldisp(result_aFME_1e9_para_best);
unlabel_ttest
test_ttest