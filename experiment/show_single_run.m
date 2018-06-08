function show_single_run(record_path, t)

%%
if ~exist(record_path, 'dir')
    error('No such directory');
end

%% load data
pca_data = fullfile(record_path, 'pca.mat');load(pca_data);
label_data = fullfile(record_path, 'label.mat');load(label_data);

%% load results
ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best2.mat');load(ffme_data_1_1e9_para_best);
%effme_data_1e9_para_best = fullfile(record_path, 'result_efFME_1e9_para_best.mat');load(effme_data_1e9_para_best);
afme_data_1e9_para_best = fullfile(record_path, 'result_aFME_1e9_para_best.mat');load(afme_data_1e9_para_best);
agr_data_para_best = fullfile(record_path, 'result_AGR_para_best.mat');load(agr_data_para_best);
eagr_data_para_best = fullfile(record_path, 'result_EAGR_para_best.mat');load(eagr_data_para_best);
mmlp_data_para = fullfile(record_path, 'result_MMLP_min_para.mat');load(mmlp_data_para);
mmlp_data_para = fullfile(record_path, 'result_MMLP_max_para.mat');load(mmlp_data_para);
if result_MMLP_min_para{1}.best_train_accuracy(1) >= result_MMLP_max_para{1}.best_train_accuracy(1)
    result_MMLP_para = result_MMLP_min_para;
    MMLP_graph = 'min';
else
    result_MMLP_para = result_MMLP_max_para;
    MMLP_graph = 'max';
end
mtc_data_para = fullfile(record_path, 'result_MTC_para.mat');load(mtc_data_para);
nn_data_para = fullfile(record_path, 'result_NN_para.mat');load(nn_data_para);
laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');load(laprls_data2_para_best);

celldisp(result_AGR_para_best);
celldisp(result_EAGR_para_best);
celldisp(result_MMLP_para);
celldisp(result_MTC_para);
celldisp(result_NN_para);
celldisp(result_LapRLS2_para_best);
celldisp(result_fastFME1_1e9_para_best);
celldisp(result_aFME_1e9_para_best);

%% load graphs
ag_data = fullfile(record_path, 'ag.mat');load(ag_data);
eag_data = fullfile(record_path, 'eag.mat');load(eag_data);
emin_data = fullfile(record_path, 'E_min.mat');load(emin_data);
emax_data = fullfile(record_path, 'E_max.mat');load(emax_data);

%% show ground truth
close all;
gs=12;
gscatter(X_train(1,:)', X_train(2,:)', Y_train, 'rc', 'x.', gs, gs, 'off');hold on;
[r, c] = find(label{1}(:,t));
plot(X_train(1,r(find(Y_train(r)==1)))', X_train(2,r(find(Y_train(r)==1)))', 'kx', 'MarkerSize', 20, 'LineWidth', 3);
hold on;
plot(X_train(1,r(find(Y_train(r)==2)))', X_train(2,r(find(Y_train(r)==2)))', 'bo', 'MarkerSize', 8, 'LineWidth', 8);
axis off;
print(gcf,'-dpng',fullfile(record_path, 'gt.png'));

%%
% run AGR
label_ind = find(label{1}(:,t));
[F, ~, e] = AnchorGraphReg(B, rL, Y_train', label_ind, ...
    result_AGR_para_best{1}.best_para);
F1 = F*diag(sum(F).^-1);
[~, F] = max(F1,[],2);
        
figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'agr.png'));

% run EAGR
best_beta = result_EAGR_para_best{1}.best_id(1)
[~, F] = EAGReg(Z{best_beta}, rLz{best_beta}, Y_train', label_ind, ...
    result_EAGR_para_best{1}.best_para(2));

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'eagr.png'));

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
p.mu = result_fastFME1_1e9_para_best{1}.best_train_para(1);
p.gamma = result_fastFME1_1e9_para_best{1}.best_train_para(2);
[W, b, F_train] = fastFME_semi(X_train, B, Y, p, true);
% F_train = F_train*diag(sum(F_train).^-1);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'ffme.png'));

% run efFME
% p.mu = result_efFME_1e9_para_best{1}.best_train_para(1);
% p.gamma = result_efFME_1e9_para_best{1}.best_train_para(2);
% [~, ~, F_train] = fastFME_semi(X_train, Z{best_beta}, Y, p);
% [~, F] = max(F_train,[],2);
% 
% figure;
% gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
% axis off;
% print(gcf,'-dpng',fullfile(record_path, 'effme.png'));

% run aFME
p.mu = result_aFME_1e9_para_best{1}.best_train_para(1);
p.gamma = result_aFME_1e9_para_best{1}.best_train_para(2);
[~, ~, F_train] = aFME_semi(anchor, Z{best_beta}, rLz{best_beta}, Y, p, true);
% F_train = F_train*diag(sum(F_train).^-1);
[~, F] = max(F_train,[],2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'afme.png'));

% run nn
unlabel_ind = find(~label{1}(:,t));
label_ind = find(label{1}(:,t));
F = flann_NN(X_train(:,label_ind), Y_train(label_ind), ...
        X_train(:,unlabel_ind));
    
figure;
gscatter(X_train(1,unlabel_ind)', X_train(2,unlabel_ind)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'nn.png'));

% run MMLP
label_index = find(label{1}(:,t));
if strcmp(MMLP_graph, 'min')
    [F, ~, ~, ~, ~] = mmlp(E_min, X_train, Y_train, label_index);
else
    [F, ~, ~, ~, ~] = mmlp(E_max, X_train, Y_train, label_index);
end

figure;
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'mmlp.png'));

% run MTC
% calculate edges
s = result_MTC_para{1}.best_s;
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
gscatter(X_train(1,:)', X_train(2,:)', F, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'mtc.png'));

% run LapRLS
% compute laplacian matrix
s = result_LapRLS2_para_best{1}.best_train_para(1);
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

p.gammaA = result_LapRLS2_para_best{1}.best_train_para(2);
p.gammaI = result_LapRLS2_para_best{1}.best_train_para(3);
unlabel_ind = find(~label{1}(:,t));
label_ind = find(label{1}(:,t));
[W, b] = LapRLS(X_train, Y_train, L, label_ind, p.gammaA, p.gammaI);
F_train = X_train' * W + ones(size(X_train, 2), 1) * b';
[~, F] = max(F_train, [], 2);

figure;
gscatter(X_train(1,:)', X_train(2,:)', F);%, 'rc', 'x.', gs, gs, 'off');
axis off;
print(gcf,'-dpng',fullfile(record_path, 'laprls.png'));                
