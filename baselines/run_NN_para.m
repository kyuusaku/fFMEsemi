function result_NN = run_NN_para(X_train, Y_train, X_test, Y_test, label)

%%
fprintf('******** Runing NN ***********\n');

np = numel(label);
result_NN = cell(np, 1);
for i = 1 : np
    [err, best_train, best_test, average_time] = ...
        iner_run_NN(X_train, Y_train, X_test, Y_test, label{i});
    result.accuracy = 100*(1-err);
    result.best_train_accuracy = [100*(1-best_train(1,1)), 100*best_train(1,2)];
    result.best_test_accuracy = [100*(1-best_test(1,1)), 100*best_test(1,2)];
    result.average_time = average_time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_NN{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [err, best_train, best_test, average_time] = ...
        iner_run_NN(X_train, Y_train, X_test, Y_test, label)
iter = size(label, 2);
err = zeros(iter, 2);
time = zeros(iter, 1);
for t = 1 : iter
    tic;
    unlabel_ind = find(~label(:,t));
    label_ind = find(label(:,t));
    predictions = flann_NN(X_train(:,label_ind), Y_train(label_ind), ...
        X_train(:,unlabel_ind));
    time(t) = toc;    
    err(t, 1) = mean(double(predictions ~= Y_train(unlabel_ind)));
    
    predictions = flann_NN(X_train(:,label_ind), Y_train(label_ind), X_test);
    err(t, 2) = mean(double(predictions ~= Y_test));
    % verbose
    fprintf('run_NN: t = %d, Unlabel = %f, Test = %f\n', t, 100*(1-err(t, 1)), 100*(1-err(t,2)));
end
mean_err = mean(err);
std_err = std(err);
best_train = [mean_err(1) std_err(1)];
best_test = [mean_err(2) std_err(2)];
average_time = mean(time);
end