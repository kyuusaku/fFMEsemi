%% effect of hard linear constraint

test_type = 'non_linear';

%% para
save_path = 'result/test/hard_linear_constraint';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, test_type);
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
para.knn = 3;
para.beta = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
save(fullfile(record_path, 'para.mat'), 'para');

%% random data
if strcmp(test_type, 'non_linear')
    a = unifrnd(-10, 10, 3000, 1);
    b = unifrnd(-1, 1, 3000, 1);
    c = unifrnd(-3, 3, 1000, 1);
    d = unifrnd(-1, 1, 1000, 1);
    fea = [a', d'+10; b'+2, c']; 
    fea = [fea, -fea - 2];
    gnd = ones(8000, 1);
    gnd(4001:8000) = 2;
end

%%
data = fullfile(record_path, 'data.mat');
if ~exist(data, 'file')
    X_train = fea(:, 1:2:8000);
    Y_train = gnd(1:2:8000);
    X_test = fea(:, 2:2:8000);
    Y_test = gnd(2:2:8000);

    label = false(4000, 1);
    label([750,2250]) = true;
    gs=12;
    gscatter(X_train(1,:)', X_train(2,:)', Y_train, 'rc', 'x.', gs, gs, 'off');
    hold on;
    [r, c] = find(label);
    plot(X_train(1,r(1))', X_train(2,r(1))', 'kx', 'MarkerSize', 20, 'LineWidth', 3);
    hold on;
    plot(X_train(1,r(2))', X_train(2,r(2))', 'bo', 'MarkerSize', 8, 'LineWidth', 8);
    axis off;
    print(gcf,'-dpng',fullfile(record_path, 'r-gnd.png'));
    % save
    save(data, 'X_train', 'Y_train', 'X_test', 'Y_test', 'label');
else
    load(data);
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

%% run fast FME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best2.mat');
if ~exist(ffme_data_1_1e9_para_best, 'file')
    result_fastFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, B, {label}, ...
        1e9, mu, gamma);
    save(ffme_data_1_1e9_para_best, 'result_fastFME1_1e9_para_best');
else
    load(ffme_data_1_1e9_para_best);
end
celldisp(result_fastFME1_1e9_para_best);

%% draw fFME
unlabel_ind = find(~label);
label_ind = find(label);
n = size(X_train, 2);
class = unique(Y_train);
n_class = numel(class);
Y = zeros(n, n_class);
for cc = 1 : n_class
    cc_ind = find(Y_train(label_ind) == cc);
    Y(label_ind(cc_ind),cc) = 1;
end
Y = sparse(Y);
p.ul = 1e9;
p.uu = 0;
p.mu = result_fastFME1_1e9_para_best{1}.best_train_para(1);
p.gamma = result_fastFME1_1e9_para_best{1}.best_train_para(2);
[W, b, F_train] = fastFME_semi(X_train, B, Y, p);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'r-ffme.png'));

%% run efFME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
best_beta = result_EAGR_para_best{1}.best_id(1)
effme_data_1_1e9_para_best = fullfile(record_path, 'result_efFME1_1e9_para_best2.mat');
if ~exist(effme_data_1_1e9_para_best, 'file')
    result_efFME1_1e9_para_best = run_fastFME_semi_para(X_train, Y_train, X_test, Y_test, ...
        Z{best_beta}, {label}, 1e9, mu, gamma);
    save(effme_data_1_1e9_para_best, 'result_efFME1_1e9_para_best');
else
    load(effme_data_1_1e9_para_best);
end
celldisp(result_efFME1_1e9_para_best);

%% draw efFME
p.mu = result_efFME1_1e9_para_best{1}.best_train_para(1);
p.gamma = result_efFME1_1e9_para_best{1}.best_train_para(2);
[W, b, F_train] = fastFME_semi(X_train, Z{best_beta}, Y, p);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'r-effme.png'));

%% run aFME
mu = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gamma = mu;
best_beta = result_EAGR_para_best{1}.best_id(1)
afme_data_1e9_para_best = fullfile(record_path, 'result_aFME_1e9_para_best.mat');
if ~exist(afme_data_1e9_para_best, 'file')
    result_aFME_1e9_para_best = run_aFME_semi_para(X_train, Y_train, X_test, Y_test, anchor, ...
        Z{best_beta}, rLz{best_beta}, {label}, 1e9, mu, gamma);
    save(afme_data_1e9_para_best, 'result_aFME_1e9_para_best');
else
    load(afme_data_1e9_para_best);
end
celldisp(result_aFME_1e9_para_best);

%% draw aFME
p.mu = result_aFME_1e9_para_best{1}.best_train_para(1);
p.gamma = result_aFME_1e9_para_best{1}.best_train_para(2);
[~, ~, F_train] = aFME_semi(anchor, Z{best_beta}, rLz{best_beta}, Y, p);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'r-afme.png'));

%% E_max
emax_data = fullfile(record_path, 'E_max.mat');
if ~exist(emax_data, 'file')
    [E_max, mmlp_gr_time_max] = knn_graph_max(X_train, para.K+1);
    save(emax_data, 'E_max', 'mmlp_gr_time_max');
else
    load(emax_data);
end

%% run LapRLS/L
best_s = []; best_gammaA = []; best_gammaI = [];
laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');
if ~exist(laprls_data2_para_best, 'file')
    result_LapRLS2_para_best = run_LapRLS2_para(X_train, Y_train, X_test, Y_test, E_max, {label}, ...
        best_s, best_gammaA, best_gammaI);
    save(laprls_data2_para_best, 'result_LapRLS2_para_best');
else
    load(laprls_data2_para_best);
end

%% draw LapRLS
% compute laplacian matrix
k = 10;
S = E_max;
s = 1e-5/k;
[ii, jj, ss] = find(S);
[mm, nn] = size(S);
s_mean = mean(ss);
para_t = - full(s_mean) ./ log(s);
W = sparse(ii, jj, exp(-ss ./ para_t), mm, nn);
W(isnan(W)) = 0; W(isinf(W)) = 0;

D = spdiags(sum(W, 2), 0, nn, nn);
L = D - W;
D(isnan(D)) = 0; D(isinf(D)) = 0;
L(isnan(L)) = 0; L(isinf(L)) = 0;

p.gammaA = 1e-9;
p.gammaI = 1e-6;
unlabel_ind = find(~label);
label_ind = find(label);
[W, b] = LapRLS(X_train, Y_train, L, label_ind, p.gammaA, p.gammaI);
F_train = X_train' * W + ones(size(X_train, 2), 1) * b';
[~, F] = max(F_train, [], 2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'r-laprls.png'));  