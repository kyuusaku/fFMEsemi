%% effect of hard linear constraint

test_type = 'linear';

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
if strcmp(test_type, 'linear')
    a = unifrnd(-5, 5, 4000, 1);
    b = unifrnd(-1, 1, 4000, 1);
    fea = [a', b'; b' + 8, a']; 
    gnd = ones(8000, 1);
    gnd(4001:8000) = 2;
end
gscatter(fea(1,:)', fea(2,:)', gnd, 'rc', 'x.', gs, gs, 'off');
% axis off;
print(gcf,'-dpng',fullfile(record_path, 'data.png'));

%%
data = fullfile(record_path, 'data.mat');
if ~exist(data, 'file')
    X_train = fea(:, 1:2:8000);
    Y_train = gnd(1:2:8000);
    X_test = fea(:, 2:2:8000);
    Y_test = gnd(2:2:8000);

    label = false(4000, 1);
    label([1000,3000]) = true;
    gs=12;
    gscatter(X_train(1,:)', X_train(2,:)', Y_train, 'rc', 'x.', gs, gs, 'off');
    hold on;
    [r, c] = find(label);
    plot(X_train(1,r(1))', X_train(2,r(1))', 'kx', 'MarkerSize', 20, 'LineWidth', 3);
    hold on;
    plot(X_train(1,r(2))', X_train(2,r(2))', 'bo', 'MarkerSize', 8, 'LineWidth', 8);
    axis off;
    print(gcf,'-dpng',fullfile(record_path, 'gnd.png'));
    % save
    save(data, 'X_train', 'Y_train', 'X_test', 'Y_test');
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
p.mu = 1e-3;
p.gamma = 1e9;
[W, b, F_train] = fastFME_semi(X_train, B, Y, p);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'ffme.png'));

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
s = 1e-9/k;
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
p.gammaI = 1e-9;
unlabel_ind = find(~label);
label_ind = find(label);
[W, b] = LapRLS(X_train, Y_train, L, label_ind, p.gammaA, p.gammaI);
F_train = X_train' * W + ones(size(X_train, 2), 1) * b';
[~, F] = max(F_train, [], 2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'laprls.png'));  