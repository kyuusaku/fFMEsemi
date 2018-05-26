function result_fastFME = run_fastFME_semi_para(...
    X_train, Y_train, X_test, Y_test, B, label, ul, mu, gamma, class_norm)

%% default parameters
if ~exist('ul', 'var') || isempty(ul)
    ul = 1;
end
uu = 0;
if ~exist('mu', 'var') || isempty(mu)
    mu = [1e-9;1e-6;1e-3;1;1e3;1e6;1e9];
end
if ~exist('gamma', 'var') || isempty(gamma)
    gamma = [1e-9;1e-6;1e-3;1;1e3;1e6;1e9];
end

%%
fprintf('******** Runing fast FME semi ***********\n');

class = unique(Y_train);

np = numel(label);
result_fastFME = cell(np, 1);
for i = 1 : np
    [errs, best_train, best_test, average_time] = ...
        iner_run_fastFME_semi(B, X_train, Y_train, X_test, Y_test, class, ...
        label{i}, mu, gamma, ul, uu, class_norm);
    result.accuracy = 100*(1 - errs);
    result.best_train_accuracy = [100*(1-best_train(1,1)), 100*best_train(1,2)];
    result.best_train_para = [best_train(2,1), best_train(2,2)];
    result.best_train_para_id = best_train(3,:);
    result.best_test_accuracy = [100*(1-best_test(1,1)), 100*best_test(1,2)];
    result.best_test_para = [best_test(2,1), best_test(2,2)];
    result.best_test_para_id = best_test(3,:);
    result.average_time = average_time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_fastFME{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, best_train, best_test, average_time] = iner_run_fastFME_semi(B, ...
    X_train, Y_train, X_test, Y_test, class, label, mu, gamma, ul, uu, class_norm)
n = size(X_train, 2);
n_class = numel(class);
iter = size(label, 2);
n_mu = numel(mu);
n_gamma = numel(gamma);
errs = zeros(n_mu, n_gamma, iter, 2);
times = zeros(n_mu, n_gamma, iter);
for pmu = 1 : n_mu
    for pgamma = 1 : n_gamma
        p.ul = ul;
        p.uu = uu;
        p.mu = mu(pmu);
        p.gamma = gamma(pgamma);

        for t = 1 : iter
            tic;
            unlabel_ind = find(~label(:,t));
            label_ind = find(label(:,t));
            Y = zeros(n, n_class);
            for cc = 1 : n_class
                cc_ind = find(Y_train(label_ind) == cc);
                Y(label_ind(cc_ind),cc) = 1;
            end
            Y = sparse(Y);

            [W, b, F_train] = fastFME_semi(X_train, B, Y, p, class_norm);
            times(pmu, pgamma, t) = toc;

            [~, predictions] = max(F_train, [], 2); clear F_train;
            errs(pmu, pgamma, t, 1) = mean(double(predictions(unlabel_ind) ...
                ~= Y_train(unlabel_ind)));

            F_test = X_test' * W + ones(size(X_test, 2), 1) * b';
            [~, predictions] = max(F_test, [], 2); clear F_test;
            errs(pmu, pgamma, t, 2) = mean(double(predictions ~= Y_test));
            
            % verbose
            fprintf('run_fastFME:\n');
            fprintf('--- mu = %e, gamma = %e, t = %d\n', mu(pmu), gamma(pgamma), t);
            fprintf('--- unlabel = %f, test = %f\n', ...
                100*(1-errs(pmu, pgamma, t, 1)), 100*(1-errs(pmu, pgamma, t, 2)));
        end
    end
end

% for output
errs_train = errs(:,:,:,1);
stds_train = std(errs_train, [], 3);
errs_train = mean(errs_train, 3);
[minError, minImu] = min(errs_train);
[minError, minIgamma] = min(minError);
minImu = minImu(minIgamma);
best_train = [minError, stds_train(minImu, minIgamma); ...
    mu(minImu), gamma(minIgamma); minImu, minIgamma];

errs_test = errs(:,:,:,2);
stds_test = std(errs_test, [], 3);
errs_test = mean(errs_test, 3);
[minError, minImu] = min(errs_test);
[minError, minIgamma] = min(minError);
minImu = minImu(minIgamma);
best_test = [minError, stds_test(minImu, minIgamma); ...
    mu(minImu), gamma(minIgamma); minImu, minIgamma];

average_time = mean(times(:));

end