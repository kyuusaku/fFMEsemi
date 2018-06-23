function result_EAGR = run_EAGR_para(Y_train, Z, rLz, label, gamma, para)

%% default parameter
gamma_para = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
if exist('gamma', 'var') && ~isempty(gamma)
    gamma_para = gamma;
end

%%
fprintf('******** Runing EAGR ***********\n');

np = numel(label);
result_EAGR = cell(np, 1);
for i = 1 : np
    [errs, err, v, best_para, best_id, time] = iner_run_EAGR(Z, rLz, Y_train, label{i}, gamma_para, para);
    result.accuracy = 100*(1-errs);
    result.best_train_accuracy = [100*(1-err), 100*v];
    result.best_para = best_para;
    result.best_id = best_id;
    result.average_time = time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_EAGR{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, err, v, best_para, best_id, time] = iner_run_EAGR(Z, rLz, Y_train, label, gamma_para, para)
n_beta = numel(para.beta);
n_gamma = numel(gamma_para);
iter = size(label, 2);
errs = zeros(n_beta, n_gamma, iter);
time = zeros(n_beta, n_gamma, iter);
for pbeta = 1 : n_beta
    for pgamma = 1 : n_gamma
        for t = 1 : iter
            label_ind = find(label(:,t));
            tic;
            [acc] = EAGReg(Z{pbeta}, rLz{pbeta}, Y_train', label_ind, gamma_para(pgamma), para.classnorm);
            time(pbeta, pgamma, t) = toc;
            errs(pbeta, pgamma, t) = 1-acc;
            % verbose
            fprintf('run_EAGR: beta = %e, gamma = %e, t = %d, accuracy = %f\n', ...
                para.beta(pbeta), gamma_para(pgamma), t, 100*acc);
        end
    end
end
err = mean(errs, 3);
v = std(errs, [], 3);
[err, minIgamma] = min(err, [], 2);
[err, minIbeta] = min(err, [], 1);
minIgamma = minIgamma(minIbeta);
v = v(minIbeta, minIgamma);
best_para = [para.beta(minIbeta), gamma_para(minIgamma)];
best_id = [minIbeta, minIgamma];
time = mean(time(:));

end