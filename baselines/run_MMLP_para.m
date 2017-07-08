function result_MMLP = run_MMLP_para(X_train, Y_train, E, label)

%%
fprintf('******** Runing MMLP ***********\n');

np = numel(label);
result_MMLP = cell(np, 1);
for i = 1 : np
    [errs, err, v, time] = iner_run_MMLP(E, X_train, Y_train, label{i}); 
    result.accuracy = 100 - errs;
    result.best_train_accuracy = [(100-err), v];
    result.average_time = time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_MMLP{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, err, v, time] = iner_run_MMLP(E, X_train, Y_train, label)
iter = size(label, 2);
errs = zeros(iter, 1);
time = zeros(iter, 1);
for t = 1 : iter
    label_index = find(label(:,t));
    [~, errs(t), ~, ~, time(t)] = mmlp(E, X_train, Y_train, label_index);
    % verbose
    fprintf('run_MMLP: t = %d, accuracy = %f\n', t, 100 - errs(t));
end
err = mean(errs);
v = std(errs);
time = mean(time);
end