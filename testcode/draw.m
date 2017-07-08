%% draw gaussian
d=1; std=0.07;
save_path = 'result/test/semi';
record_path = fullfile(save_path, ['record_2017-gaussian-' num2str(d) '-' num2str(std)]);
if ~exist(record_path, 'dir')
    error('No such directory');
end

% load data & results
pca_data = fullfile(record_path, 'pca.mat');
load(pca_data);
label_data = fullfile(record_path, 'label.mat');
load(label_data);
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best2.mat');
load(ffme_data_1_1e9_para_best);
agr_data_para_best = fullfile(record_path, 'result_AGR_para_best.mat');
load(agr_data_para_best);
mmlp_data_para = fullfile(record_path, 'result_MMLP_min_para.mat');
load(mmlp_data_para);
mmlp_data_para = fullfile(record_path, 'result_MMLP_max_para.mat');
load(mmlp_data_para);
if result_MMLP_min_para{1}.best_train_accuracy(1) >= result_MMLP_max_para{1}.best_train_accuracy(1)
    result_MMLP_para = result_MMLP_min_para;
else
    result_MMLP_para = result_MMLP_max_para;
end
mtc_data_para = fullfile(record_path, 'result_MTC_para.mat');
load(mtc_data_para);
nn_data_para = fullfile(record_path, 'result_NN_para.mat');
load(nn_data_para);
laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');
load(laprls_data2_para_best);

% load graphs
ag_data = fullfile(record_path, 'ag.mat');
load(ag_data);
emin_data = fullfile(record_path, 'E_min.mat');
load(emin_data);
emax_data = fullfile(record_path, 'E_max.mat');
load(emax_data);


%%
t=4;
gscatter(X_train(1,:)', X_train(2,:)', Y_train, 'rc', 'x.');
hold on;
[r, c] = find(label{1}(:,t));
plot(X_train(1,r(1))', X_train(2,r(1))', 'kx', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(X_train(1,r(2))', X_train(2,r(2))', 'bo', 'MarkerSize', 6, 'LineWidth', 6);
axis off;
print(gcf,'-dpng',fullfile(record_path, 'gt.png'));

%%
% run AGR
label_ind = find(label{1}(:,t));
[F, ~, e] = AnchorGraphReg(B, rL, Y_train', label_ind, 1);
F1 = F*diag(sum(F).^-1);
[~, F] = max(F1,[],2);
        
figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'agr.png'));

% run fFME
unlabel_ind = find(~label{1}(:,t));
label_ind = find(label{1}(:,t));
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
p.mu = 1e24;
p.gamma = 1e-9;
[W, b, F_train] = fastFME_semi(X_train, B, Y, p);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'ffme.png'));

% run nn
unlabel_ind = find(~label{1}(:,t));
label_ind = find(label{1}(:,t));
F = flann_NN(X_train(:,label_ind), Y_train(label_ind), ...
        X_train(:,unlabel_ind));
    
figure;
gscatter(X_train(1,unlabel_ind)', X_train(2,unlabel_ind)', F, 'rc', 'x.');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'nn.png'));

% run MMLP
label_index = find(label{1}(:,t));
[F, ~, ~, ~, ~] = mmlp(E_min, X_train, Y_train, label_index);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'mmlp.png'));

% run MTC
% calculate edges
k = 10;
s = 1e-5/k;
S = E_max;
[ii, jj, ss] = find(S);
[mm, nn] = size(S);
s_mean = mean(ss);
para_t = - full(s_mean) ./ log(s);
W = sparse(ii, jj, exp(-ss ./ para_t), mm, nn);
W(isnan(W)) = 0; W(isinf(W)) = 0;
[idx1, idx2] = find(W~=0);
edges = [idx1 idx2 W(idx1+(idx2-1).*n)];

label_ind = find(label{1}(:,t));
unlabel_ind = find(~label{1}(:,t));
% construct Y    
Y = zeros(n, 1)-1;
Y(label_ind) = Y_train(label_ind) - 1;
% compute F
F = mtc_matlab(full(edges), n, Y, n_class, 0, 1);
F = F + 1;

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'mtc.png'));

% run LapRLS
% compute laplacian matrix
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

p.gammaA = 1e-6;
p.gammaI = 1e-9;
unlabel_ind = find(~label{1}(:,t));
label_ind = find(label{1}(:,t));
[W, b] = LapRLS(X_train, Y_train, L, label_ind, p.gammaA, p.gammaI);
F_train = X_train' * W + ones(size(X_train, 2), 1) * b';
[~, F] = max(F_train, [], 2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'laprls.png'));                