function two_step_experiment(dataset,o,varargin)
% two step experiment
% Step 1: AGR or EAGR obtain the optimal prediction
% Step 2: Use the prediction to train RidgeRegression and get the result of
%         test set

% parse inputs
p = parse_inputs();
parse(p,dataset,o,varargin{:});

% path
%data_name = get_data_name(p.Results.dataset);
save_path = ['result/' p.Results.dataset '/semi-a' num2str(p.Results.anchorNumber)];
if ~exist(save_path, 'dir')
    error([save_path ' not exist']);
end
record_path = fullfile(save_path, ['record-' num2str(p.Results.o)]);
if ~exist(record_path, 'dir')
    error([record_path ' not exist']);
end
load(fullfile(record_path, 'para.mat'));

% data
pca_data = fullfile(save_path, 'pca.mat');
if ~exist(pca_data, 'file')
    error([pca_data ' not exist']);
else
    load(pca_data);
end

label_data = fullfile(record_path, 'label.mat');
if ~exist(label_data, 'file')
    error([label_data ' not exist']);
else
    load(label_data);
end

ag_data = fullfile(save_path, 'ag.mat');
if ~exist(ag_data, 'file')
    error([ag_data ' not exist']);
else
    load(ag_data);
end

eag_data = fullfile(save_path, 'eag.mat');
if ~exist(eag_data, 'file')
    error([eag_data ' not exist']);
else
    load(eag_data);
end

% load result
agr_data_para_best = fullfile(record_path, 'result_AGR_para_best.mat');
if ~exist(agr_data_para_best, 'file')
    error([agr_data_para_best ' not exist']);
else
    load(agr_data_para_best);
end

eagr_data_para_best = fullfile(record_path, 'result_EAGR_para_best.mat');
if ~exist(eagr_data_para_best, 'file')
    error([eagr_data_para_best ' not exist']);
else
    load(eagr_data_para_best);
end

% run 
%gammaA = [1e-24;1e-21;1e-18;1e-15;1e-12;1e-9;1e-6;1e-3;1;1e3;1e6;1e9;1e12;1e15;1e18;1e21;1e24];
gammaA = linspace(0.001, 10, 20);

best_gamma = result_AGR_para_best{1}.best_para;

result_AGR_RS = cell(1, 1);
[errs, best_accuracy, best_para, best_id, time] = ...
        iner_run_AGR_RidgeRegression(X_train, Y_train, X_test, Y_test, ...
        label{1}, gammaA, B, rL, best_gamma, para.classnorm);
result.accuracy = 100 * (1-errs);
result.best_train_accuracy = [100*(1-best_accuracy(1,1)), 100*best_accuracy(1,2)];
result.best_train_para = best_para(1,:);
result.best_train_para_id = best_id(1,:);
result.best_test_accuracy = [100*(1-best_accuracy(2,1)), 100*best_accuracy(2,2)];
result.best_test_para = best_para(2,:);
result.best_test_para_id = best_id(2,:);
result.average_time = time;
result.p = sum(label{1});
result.p = result.p(1);
result_AGR_RS{1} = result;


celldisp(result_AGR_para_best);
celldisp(result_AGR_RS);

end

%%
function [errs, best_accuracy, best_para, best_id, time] = ...
    iner_run_AGR_RidgeRegression(X_train, Y_train, X_test, Y_test, label, gammaA, ...
                                 B, rL, best_gamma, class_norm)
iter = size(label, 2);
n_gammaA = numel(gammaA);
errs = zeros(n_gammaA, iter, 2);
times = zeros(n_gammaA, iter);         
% solve ridge regression    
for pgammaA = 1 : n_gammaA
    p_gammaA = gammaA(pgammaA);
    for t = 1 : iter
        tic;
        unlabel_ind = find(~label(:,t));
        label_ind = find(label(:,t));
        
        [F, ~, ~, ~] = AnchorGraphReg(B, rL, Y_train', label_ind, best_gamma, class_norm);
        if class_norm
            F1 = F*diag(sum(F).^-1);
        else
            F1 = F;
        end
        [~, F] = max(F1,[],2);
        
        [W, b] = RS(X_train, F, label_ind, p_gammaA);
        times(pgammaA, t) = toc;

        F_train = X_train' * W + ones(size(X_train, 2), 1) * b';
        [~, predictions] = max(F_train, [], 2); clear F_train;
        errs(pgammaA, t, 1) = ...
            mean(double(predictions(unlabel_ind) ~= Y_train(unlabel_ind)));

        F_test = X_test' * W + ones(size(X_test, 2), 1) * b';
        [~, predictions] = max(F_test, [], 2); clear F_test;
        errs(pgammaA, t, 2) = mean(double(predictions ~= Y_test));
                
        % verbose
        fprintf('run_RS:\n');
        fprintf('--- gammaA = %e, t = %d\n', gammaA(pgammaA), t);
        fprintf('--- unlabel = %f, test = %f\n', ...
                100*(1-errs(pgammaA, t, 1)), ...
                100*(1-errs(pgammaA, t, 2)));
    end
end

% for output
errs_train = errs(:,:,1);
stds_train = std(errs_train, [], 2);
errs_train = mean(errs_train, 2);
[errs_train, minIgammaA_train] = min(errs_train, [], 1);
std_train = stds_train(minIgammaA_train);

errs_test = errs(:,:,2);
stds_test = std(errs_test, [], 2);
errs_test = mean(errs_test, 2);
[errs_test, minIgammaA_test] = min(errs_test, [], 1);
std_test = stds_test(minIgammaA_test);

best_accuracy = [errs_train, std_train; errs_test, std_test];
best_para = [gammaA(minIgammaA_train); gammaA(minIgammaA_test)];
best_id = [minIgammaA_train; minIgammaA_test];

time = mean(times(:));

end

%%
function [W, b] = RS(X_train, Y_train, label_ind, gammaA)
nFea = size(X_train, 1);
% construct the labeled matrix
feaLabel = X_train(:, label_ind);
gndLabel = Y_train(label_ind);
classLabel = unique(gndLabel);
nClass = numel(classLabel);
nLabel = numel(gndLabel);
YLabel = zeros(nLabel, nClass);
for i = 1 : nClass
    YLabel(gndLabel == i, classLabel(i)) = 1;
end
% compute W
Xl = bsxfun(@minus, feaLabel, mean(feaLabel,2));
W = (Xl * Xl' + gammaA * nLabel .* eye(nFea)) \ (Xl * YLabel);
b = 1/nLabel*(sum(YLabel,1)' - W'*(feaLabel*ones(nLabel,1)));

end

