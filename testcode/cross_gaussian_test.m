function cross_gaussian_test(d)

%% random data
[fea, gnd] = cross_gaussian(5000, d);
fea = fea';
gscatter(fea(1,:)', fea(2,:)', gnd);

%% para
save_path = 'result/test/semi';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, ['record_20161226-crossgaussian-' num2str(d)]);
if ~exist(record_path, 'dir')
    mkdir(record_path);
end

para.iter = 20;
para.type = 'equal';
para.K = 10;
para.p = 1;% label number of each class
para.s = 3; % anchor
para.cn = 10;
para.num_anchor = 1000;
save(fullfile(record_path, 'para.mat'), 'para');

%% random data
% std = [0,1,2,3,4,5]
[fea, gnd] = cross_gaussian(5000, d);
fea = fea';
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

%% run fast FME
best_mu = []; best_gamma = [];
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best.mat');
if ~exist(ffme_data_1_1e9_para_best, 'file')
    result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, label, ...
        1e9, best_mu, best_gamma);
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

%% run GFHF
best_s = [];
gfhf_data_para_best = fullfile(record_path, 'result_GFHF_para_best.mat');
if ~exist(gfhf_data_para_best, 'file')
    result_GFHF_para_best = run_GFHF_para(Y_train, E_max, label, best_s);
    save(gfhf_data_para_best, 'result_GFHF_para_best');
else
    load(gfhf_data_para_best);
end

%% run LGC
best_s = [];
lgc_data_para_best = fullfile(record_path, 'result_LGC_para_best.mat');
if ~exist(lgc_data_para_best, 'file')
    result_LGC_para_best = run_LGC_para(Y_train, E_max, label, best_s);
    save(lgc_data_para_best, 'result_LGC_para_best');
else
    load(lgc_data_para_best);
end

%% run FME
best_s = []; best_mu = []; best_gamma = [];
fme_data_1_1_para_best = fullfile(record_path, 'result_FME1_1_para_best.mat');
if ~exist(fme_data_1_1_para_best, 'file')
    result_FME1_1_para_best = run_FME_semi_para(X_train, Y_train, X_test, Y_test, E_max, label, ...
        1, best_mu, best_gamma, best_s);
    save(fme_data_1_1_para_best, 'result_FME1_1_para_best');
else
    load(fme_data_1_1_para_best);
end


%% ttest
% Unlabel ttest 1=PVM, 2=AGR, 3=MMLP, 4=1NN, 5=LapRLS, 6=fastFME
X_GFHF = result_GFHF_para_best{1}.accuracy(result_GFHF_para_best{1}.best_id, :);
X_LGC = result_LGC_para_best{1}.accuracy(result_LGC_para_best{1}.best_id, :);
X_AGR = result_AGR_para_best{1}.accuracy(result_AGR_para_best{1}.best_id, :);
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
X_unlabel = {X_GFHF; X_LGC; X_AGR; X_MMLP; X_MTC; X_NN_u; X_LapRLS_u; X_fastFME_u};
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
X_test = {X_NN_t; X_LapRLS_t; X_fastFME_t};
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
celldisp(result_GFHF_para_best);
celldisp(result_LGC_para_best);
celldisp(result_AGR_para_best);
celldisp(result_MMLP_para);
celldisp(result_MTC_para);
celldisp(result_NN_para);
celldisp(result_LapRLS2_para_best);
celldisp(result_fastFME1_1e9_para_best);
celldisp(result_FME1_1_para_best);
unlabel_ttest
test_ttest