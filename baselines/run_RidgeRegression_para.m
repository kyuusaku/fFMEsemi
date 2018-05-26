function result_RS = run_RidgeRegression_para(X_train, Y_train, X_test, Y_test, ...
    label, A)

%% default parameters
% gammaA = [1e-9;1e-6;1e-3;1;1e3;1e6;1e9];
% gammaA = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
gammaA = linspace(0.001, 10, 20);
if exist('A', 'var') && ~isempty(A)
    gammaA = A;
end

%%
fprintf('******** Runing Ridge Regression ***********\n');

np = numel(label);
result_RS = cell(np, 1);
for i = 1 : np      
    [errs, best_accuracy, best_para, best_id, time] = ...
        iner_run_RidgeRegression(X_train, Y_train, X_test, Y_test, ...
        label{i}, gammaA);
    result.accuracy = 100 * (1-errs);
    result.best_train_accuracy = [100*(1-best_accuracy(1,1)), 100*best_accuracy(1,2)];
    result.best_train_para = best_para(1,:);
    result.best_train_para_id = best_id(1,:);
    result.best_test_accuracy = [100*(1-best_accuracy(2,1)), 100*best_accuracy(2,2)];
    result.best_test_para = best_para(2,:);
    result.best_test_para_id = best_id(2,:);
    result.average_time = time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_RS{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, best_accuracy, best_para, best_id, time] = ...
    iner_run_RidgeRegression(X_train, Y_train, X_test, Y_test, label, gammaA)
iter = size(label, 2);
n_gammaA = numel(gammaA);
errs = zeros(n_gammaA, iter, 2);
times = zeros(n_gammaA, iter);         
%% solve ridge regression    
for pgammaA = 1 : n_gammaA
    p.gammaA = gammaA(pgammaA);
    for t = 1 : iter
        tic;
        unlabel_ind = find(~label(:,t));
        label_ind = find(label(:,t));
        [W, b] = RS(X_train, Y_train, label_ind, p.gammaA);
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