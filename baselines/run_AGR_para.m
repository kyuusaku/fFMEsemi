function result_AGR = run_AGR_para(Y_train, B, rL, label, gamma, class_norm)

%% default parameter
gamma_para = [1e-3;1e-2;1e-1;1;1e1;1e2;1e3];
if exist('gamma', 'var') && ~isempty(gamma)
    gamma_para = gamma;
end

%%
fprintf('******** Runing AGR ***********\n');

np = numel(label);
result_AGR = cell(np, 1);
for i = 1 : np
    [errs, err, v, best_para, best_id, time] = iner_run_AGR(B, rL, Y_train, label{i}, gamma_para, class_norm);
    result.accuracy = 100*(1-errs);
    result.best_train_accuracy = [100*(1-err), 100*v];
    result.best_para = best_para;
    result.best_id = best_id;
    result.average_time = time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_AGR{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, err, v, best_para, best_id, time] = iner_run_AGR(B, rL, Y_train, label, gamma_para, class_norm)
n_gamma = numel(gamma_para);
iter = size(label, 2);
errs = zeros(n_gamma, iter);
time = zeros(n_gamma, iter);
for pgamma = 1 : n_gamma
    for t = 1 : iter
        label_ind = find(label(:,t));
        tic;
        [~, ~, e] = AnchorGraphReg(B, rL, Y_train', label_ind, gamma_para(pgamma), class_norm);
        time(pgamma, t) = toc;
        errs(pgamma, t) = e;
        % verbose
        fprintf('run_AGR: gamma = %e, t = %d, accuracy = %f\n', ...
            gamma_para(pgamma), t, 100*(1-e));
    end
end
err = mean(errs, 2);
v = std(errs, [], 2);
[err, minIgamma] = min(err);
v = v(minIgamma);
best_para = gamma_para(minIgamma);
best_id = minIgamma;
time = mean(time(:));

end