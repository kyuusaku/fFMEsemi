function result_LapRLS = run_LapRLS2_para(X_train, Y_train, X_test, Y_test, ...
    S, label, s, A, I, k)

%% default parameters
if ~exist('k', 'var')
    k = 10;
end

s_para = [1e-9;1e-7;1e-5;1e-3;1e-1];
if exist('s', 'var') && ~isempty(s)
    s_para = s;
end

gammaA = [1e-9;1e-6;1e-3;1;1e3;1e6;1e9];
% gammaA = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
% gammaA = linspace(0.001, 10, 20);
if exist('A', 'var') && ~isempty(A)
    gammaA = A;
end
% gammaI = [1e-9;1e-6;1e-3;1;1e3;1e6;1e9];
gammaI = gammaA;
if exist('I', 'var') && ~isempty(I)
    gammaI = I;
end

%%
fprintf('******** Runing LapRLS ***********\n');

np = numel(label);
result_LapRLS = cell(np, 1);
for i = 1 : np      
    [errs, best_accuracy, best_para, best_id, time, l_time] = ...
        iner_run_LapRLS(S, X_train, Y_train, X_test, Y_test, ...
        label{i}, s_para, gammaA, gammaI, k);
    result.accuracy = 100 * (1-errs);
    result.best_train_accuracy = [100*(1-best_accuracy(1,1)), 100*best_accuracy(1,2)];
    result.best_train_para = best_para(1,:);
    result.best_train_para_id = best_id(1,:);
    result.best_test_accuracy = [100*(1-best_accuracy(2,1)), 100*best_accuracy(2,2)];
    result.best_test_para = best_para(2,:);
    result.best_test_para_id = best_id(2,:);
    result.average_time = time;
    result.l_time = l_time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_LapRLS{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, best_accuracy, best_para, best_id, time, l_time] = ...
    iner_run_LapRLS(S, X_train, Y_train, X_test, Y_test, label, s_para, gammaA, gammaI, k)
iter = size(label, 2);
n_s = numel(s_para);
n_gammaA = numel(gammaA);
n_gammaI = numel(gammaI);
errs = zeros(n_s, n_gammaA, n_gammaI, iter, 2);
times = zeros(n_s, n_gammaA, n_gammaI, iter);
l_time = zeros(n_s, 1);
for ps = 1 : n_s
    %% compute laplacian matrix
    s = s_para(ps)/k;

    t_start = tic;
    [ii, jj, ss] = find(S);
    [mm, nn] = size(S);
    s_mean = mean(ss);
    para_t = - full(s_mean) ./ log(s);
    W = sparse(ii, jj, exp(-ss ./ para_t), mm, nn);
    W(isnan(W)) = 0; W(isinf(W)) = 0;
    w_time = toc(t_start);

    t_start = tic;
    D = spdiags(sum(W, 2), 0, nn, nn);
    L = D - W;
    D(isnan(D)) = 0; D(isinf(D)) = 0;
    L(isnan(L)) = 0; L(isinf(L)) = 0;
    l_time(ps) = toc(t_start);
    l_time(ps) = l_time(ps) + w_time;
           
    %% solve LapRLS    
    for pgammaA = 1 : n_gammaA
        for pgammaI = 1 : n_gammaI
            p.gammaA = gammaA(pgammaA);
            p.gammaI = gammaI(pgammaI);
            for t = 1 : iter
                tic;
                unlabel_ind = find(~label(:,t));
                label_ind = find(label(:,t));
                [W, b] = LapRLS(X_train, Y_train, L, label_ind, p.gammaA, p.gammaI);
                times(ps, pgammaA, pgammaI, t) = toc;

                F_train = X_train' * W + ones(size(X_train, 2), 1) * b';
                [~, predictions] = max(F_train, [], 2); clear F_train;
                errs(ps, pgammaA, pgammaI, t, 1) = ...
                    mean(double(predictions(unlabel_ind) ~= Y_train(unlabel_ind)));

                F_test = X_test' * W + ones(size(X_test, 2), 1) * b';
                [~, predictions] = max(F_test, [], 2); clear F_test;
                errs(ps, pgammaA, pgammaI, t, 2) = mean(double(predictions ~= Y_test));
                
                % verbose
                fprintf('run_LapRLS:\n');
                fprintf('--- s = %e, gammaA = %e, gammaI = %e, t = %d\n', ...
                    s_para(ps), gammaA(pgammaA), gammaI(pgammaI), t);
                fprintf('--- unlabel = %f, test = %f\n', ...
                    100*(1-errs(ps, pgammaA, pgammaI, t, 1)), ...
                    100*(1-errs(ps, pgammaA, pgammaI, t, 2)));
            end
        end
    end
end

% for output
errs_train = errs(:,:,:,:,1);
stds_train = std(errs_train, [], 4);
errs_train = mean(errs_train, 4);
[errs_train, minIgammaI_train] = min(errs_train, [], 3);
[errs_train, minIgammaA_train] = min(errs_train, [], 2);
[errs_train, minIs_train] = min(errs_train, [], 1);
minIgammaA_train = minIgammaA_train(minIs_train);
minIgammaI_train = minIgammaI_train(minIs_train, minIgammaA_train);
std_train = stds_train(minIs_train, minIgammaA_train, minIgammaI_train);

errs_test = errs(:,:,:,:,2);
stds_test = std(errs_test, [], 4);
errs_test = mean(errs_test, 4);
[errs_test, minIgammaI_test] = min(errs_test, [], 3);
[errs_test, minIgammaA_test] = min(errs_test, [], 2);
[errs_test, minIs_test] = min(errs_test, [], 1);
minIgammaA_test = minIgammaA_test(minIs_test);
minIgammaI_test = minIgammaI_test(minIs_test, minIgammaA_test);
std_test = stds_test(minIs_test, minIgammaA_test, minIgammaI_test);

best_accuracy = [errs_train, std_train; errs_test, std_test];
best_para = [s_para(minIs_train), gammaA(minIgammaA_train), gammaI(minIgammaI_train);
    s_para(minIs_test), gammaA(minIgammaA_test), gammaI(minIgammaI_test)];
best_id = [minIs_train, minIgammaA_train, minIgammaI_train;
    minIs_test, minIgammaA_test, minIgammaI_test];

time = mean(times(:));
l_time = mean(l_time(:));

end

function [W, b] = LapRLS(X_train, Y_train, L, label_ind, gammaA, gammaI)
nFea = size(X_train, 1);
% construct the labeled matrix
feaLabel = X_train(:, label_ind);
gndLabel = Y_train(label_ind);
classLabel = unique(gndLabel);
nClass = numel(classLabel);
nLabel = numel(gndLabel);
YLabel = zeros(nLabel, nClass);
for i = 1 : nClass
    YLabel(gndLabel == i, i) = 1;
end
% compute W
Xl = bsxfun(@minus, feaLabel, mean(feaLabel,2));
W = (Xl * Xl' + gammaA * nLabel .* eye(nFea) + ...
    gammaI * nLabel .* (X_train * L * X_train')) \ (Xl * YLabel);
b = 1/nLabel*(sum(YLabel,1)' - W'*(feaLabel*ones(nLabel,1)));

end