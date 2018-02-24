function result_FME = run_FME_semi_para(...
    X_train, Y_train, X_test, Y_test, S, label, ul, mu, gamma, s, k)

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

if ~exist('k', 'var')
    k = 10;
end

s_para = [1e-9;1e-7;1e-5;1e-3;1e-1];
if exist('s', 'var') && ~isempty(s)
    s_para = s;
end

%%
fprintf('******** Runing FME semi ***********\n');

class = unique(Y_train);

np = numel(label);
result_FME = cell(np, 1);
for i = 1 : np
    [errs, best_accuracy, best_para, best_id, time, l_time] = ...
        iner_run_FME_semi(S, X_train, Y_train, X_test, Y_test, class, ...
        label{i}, mu, gamma, ul, uu, s_para, k);
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
    result_FME{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, best_accuracy, best_para, best_id, time, l_time] = ...
    iner_run_FME_semi(S, X_train, Y_train, X_test, Y_test, class, ...
    label, mu, gamma, ul, uu, s_para, k)
n = size(X_train, 2);
n_class = numel(class);
iter = size(label, 2);
n_s = numel(s_para);
n_mu = numel(mu);
n_gamma = numel(gamma);
errs = zeros(n_s, n_mu, n_gamma, 2);
times = zeros(n_s, n_mu, n_gamma);
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
    
    for pmu = 1 : n_mu
        for pgamma = 1 : n_gamma
            p.ul = ul;
            p.uu = uu;
            p.mu = mu(pmu);
            p.lamda = gamma(pgamma);

            for t = 1 : iter
                tic;
                unlabel_ind = find(~label(:,t));
                label_ind = find(label(:,t));
                Y = zeros(n, n_class);
                for cc = 1 : n_class
                    cc_ind = find(Y_train(label_ind) == class(cc));
                    Y(label_ind(cc_ind),cc) = 1;
                end
                Y = sparse(Y);

                [W, b, F_train] = FME_semi(X_train, L, Y, p); clear Y;
                times(ps, pmu, pgamma, t) = toc;

                [~, predictions] = max(F_train, [], 2); clear F_train;
                errs(ps, pmu, pgamma, t, 1) = mean(double(predictions(unlabel_ind) ...
                    ~= Y_train(unlabel_ind)));

                F_test = X_test' * W + ones(size(X_test, 2), 1) * b';
                [~, predictions] = max(F_test, [], 2); clear F_test;
                errs(ps, pmu, pgamma, t, 2) = mean(double(predictions ~= Y_test));
                
                % verbose
                fprintf('run_FME:\n');
                fprintf('--- s = %e, mu = %e, gamma = %e, t = %d\n', ...
                    s_para(ps), mu(pmu), gamma(pgamma), t);
                fprintf('--- unlabel = %f, test = %f, time = %f\n', ...
                    100*(1-errs(ps, pmu, pgamma, t, 1)), ...
                    100*(1-errs(ps, pmu, pgamma, t, 2)), times(ps, pmu, pgamma, t));
            end
        end
    end
end

% for output
errs_train = errs(:,:,:,:,1);
stds_train = std(errs_train, [], 4);
errs_train = mean(errs_train, 4);
[errs_train, minIgamma_train] = min(errs_train, [], 3);
[errs_train, minImu_train] = min(errs_train, [], 2);
[errs_train, minIs_train] = min(errs_train, [], 1);
minImu_train = minImu_train(minIs_train);
minIgamma_train = minIgamma_train(minIs_train, minImu_train);
std_train = stds_train(minIs_train, minImu_train, minIgamma_train);

errs_test = errs(:,:,:,:,2);
stds_test = std(errs_test, [], 4);
errs_test = mean(errs_test, 4);
[errs_test, minIgamma_test] = min(errs_test, [], 3);
[errs_test, minImu_test] = min(errs_test, [], 2);
[errs_test, minIs_test] = min(errs_test, [], 1);
minImu_test = minImu_test(minIs_test);
minIgamma_test = minIgamma_test(minIs_test, minImu_test);
std_test = stds_test(minIs_test, minImu_test, minIgamma_test);

best_accuracy = [errs_train, std_train; errs_test, std_test];
best_para = [s_para(minIs_train), mu(minImu_train), gamma(minIgamma_train);
    s_para(minIs_test), mu(minImu_test), gamma(minIgamma_test)];
best_id = [minIs_train, minImu_train, minIgamma_train;
    minIs_test, minImu_test, minIgamma_test];

time = mean(times(:));
l_time = mean(l_time(:));

end