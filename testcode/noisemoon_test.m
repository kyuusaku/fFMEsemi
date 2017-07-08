% noisemoon data test

%% random data
data = load('./testcode/noisy_moons_2w.txt');
fea = data(:,[1,2])';
gnd = data(:,3) + 1;

gscatter(fea(1,:)', fea(2,:)', gnd);

%% para
save_path = 'result/test/semi';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, 'record_20150901-moon');
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

%% run E2
e2_data = fullfile(record_path, 'E2.mat');
if ~exist(e2_data, 'file')
    [E2, mmlp_gr_time] = knn_graph2(X_train, para.K + 1);
    save(e2_data, 'E2', 'mmlp_gr_time');
else
    load(e2_data);
end

%% compute laplacian graph
s_data = fullfile(record_path, 's.mat');
if ~exist(s_data, 'file')
    tic;S = constructS(X_train, para.K);s_time=toc;
    save(s_data, 'S', 's_time');
else
    load(s_data);
end

%% para
warning off all;

% GFHF
% load(fullfile(record_path, 'result_GFHF.mat'));
% best_s = result_GFHF{1}.best_s;
best_s = [];
gfhf_data_para_best = fullfile(record_path, 'result_GFHF_para_best.mat');
if ~exist(gfhf_data_para_best, 'file')
    result_GFHF_para_best = run_GFHF_para(Y_train, S, label, best_s);
    save(gfhf_data_para_best, 'result_GFHF_para_best');
else
    load(gfhf_data_para_best);
end

% LGC
% load(fullfile(record_path, 'result_LGC.mat'));
% best_s = result_LGC{1}.best_s;
best_s = [];
lgc_data_para_best = fullfile(record_path, 'result_LGC_para_best.mat');
if ~exist(lgc_data_para_best, 'file')
    result_LGC_para_best = run_LGC_para(Y_train, S, label, best_s);
    save(lgc_data_para_best, 'result_LGC_para_best');
else
    load(lgc_data_para_best);
end

% PVM
%load(fullfile(record_path, 'result_PVM.mat'));
%best_s = result_PVM{1}.best_para(1); best_C1 = result_PVM{1}.best_para(2);
% best_s = []; best_C1 = [];
% pvm_data_para_best = fullfile(record_path, 'result_PVM_para_best.mat');
% if ~exist(pvm_data_para_best, 'file')
%     result_PVM_para_best = run_PVM_para(X_train, Y_train, anchor, label, ...
%         best_s, best_C1);
%     save(pvm_data_para_best, 'result_PVM_para_best');
% else
%     load(pvm_data_para_best);
% end

% run AGR
% load(fullfile(record_path, 'result_AGR.mat'));
% best_gamma = result_AGR{1}.best_para;
best_gamma = [];
agr_data_para_best = fullfile(record_path, 'result_AGR_para_best.mat');
if ~exist(agr_data_para_best, 'file')
    result_AGR_para_best = run_AGR_para(Y_train, B, rL, label, best_gamma);
    save(agr_data_para_best, 'result_AGR_para_best');
else
    load(agr_data_para_best);
end

% MMLP
% load(fullfile(record_path, 'result_MMLP2.mat'));
load(fullfile(record_path, 'E2.mat'));
mmlp2_data_para = fullfile(record_path, 'result_MMLP2_para.mat');
if ~exist(mmlp2_data_para, 'file')
    result_MMLP2_para = run_MMLP_para(X_train, Y_train, E2, label);
    save(mmlp2_data_para, 'result_MMLP2_para');
else
    load(mmlp2_data_para);
end

% MTC
best_s = [];
mtc_data_para = fullfile(record_path, 'result_MTC_para.mat');
if ~exist(mtc_data_para, 'file')
    result_MTC_para = run_MTC_para(Y_train, S, label, best_s);
    save(mtc_data_para, 'result_MTC_para');
else
    load(mtc_data_para);
end

% NN
nn_data_para = fullfile(record_path, 'result_NN_para.mat');
if ~exist(nn_data_para, 'file')
    result_NN_para = run_NN_para(X_train, Y_train, X_test, Y_test, label);
    save(nn_data_para, 'result_NN_para');
else
    load(nn_data_para);
end

% LapRLS
best_s = []; best_gammaA = []; best_gammaI = [];
laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');
if ~exist(laprls_data2_para_best, 'file')
    result_LapRLS2_para_best = run_LapRLS2_para(X_train, Y_train, X_test, Y_test, S, label, ...
        best_s, best_gammaA, best_gammaI);
    save(laprls_data2_para_best, 'result_LapRLS2_para_best');
else
    load(laprls_data2_para_best);
end

% run fast FME
% load(fullfile(record_path, 'result_fastFME1_1e9.mat'));
% best_mu = [result_fastFME1_1e9{1}.best_train_para(1); result_fastFME1_1e9{1}.best_test_para(1)]; 
% best_gamma = [result_fastFME1_1e9{1}.best_train_para(2); result_fastFME1_1e9{1}.best_test_para(2)];
best_mu = []; best_gamma = [];
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best.mat');
if ~exist(ffme_data_1_1e9_para_best, 'file')
    result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, label, ...
        1e9, best_mu, best_gamma);
    save(ffme_data_1_1e9_para_best, 'result_fastFME1_1e9_para_best');
else
    load(ffme_data_1_1e9_para_best);
end

% ttest
% Unlabel ttest 1=GFHF, 2=LGC, 3=PVM, 4=AGR, 5=MMLP, 6=1NN, 7=LapRLS, 8=fastFME
X_GFHF = result_GFHF_para_best{1}.accuracy(result_GFHF_para_best{1}.best_id, :);
X_LGC = result_LGC_para_best{1}.accuracy(result_LGC_para_best{1}.best_id, :);
%X_PVM = result_PVM_para_best{1}.accuracy(result_PVM_para_best{1}.best_id(1), ...
%    result_PVM_para_best{1}.best_id(2), :); X_PVM = squeeze(X_PVM)';
X_AGR = result_AGR_para_best{1}.accuracy(result_AGR_para_best{1}.best_id, :);
X_MMLP = result_MMLP2_para{1}.accuracy';
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
X_unlabel = {X_GFHF; X_LGC; ...%X_PVM;
    X_AGR; X_MMLP; X_MTC; X_NN_u; X_LapRLS_u; X_fastFME_u};
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
% celldisp(result_PVM_para_best);
celldisp(result_AGR_para_best);
celldisp(result_MMLP2_para);
celldisp(result_MTC_para);
celldisp(result_NN_para);
celldisp(result_LapRLS2_para_best);
celldisp(result_fastFME1_1e9_para_best);
unlabel_ttest
test_ttest