function result_MTC = run_MTC_para(Y_train, S, label, s, k)

%% default parameters
if ~exist('k', 'var')
    k = 10;
end

s_para = [1e-9;1e-7;1e-5;1e-3;1e-1];
if exist('s', 'var') && ~isempty(s)
    s_para = s;
end

%%
fprintf('******** Runing MTC ***********\n');

class = unique(Y_train);

np = numel(label);
result_MTC = cell(np, 1);
for i = 1 : np
    [errs, err, v, time, l_time, best_s, best_id] = iner_run_MTC(S, ...
    Y_train, class, label{i}, s_para, k);
    result.accuracy = 100*(1-errs);
    result.best_train_accuracy = [100*(1-err), 100*v];
    result.best_s = best_s;
    result.best_id = best_id;
    result.average_time = time;
    result.l_time = l_time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_MTC{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, err, v, time, l_time, best_s, best_id] = iner_run_MTC(S, ...
    Y_train, class, label, s_para, k)
n_s = numel(s_para);
n = numel(Y_train);
n_class = numel(class);
iter = size(label, 2);
errs = zeros(n_s, iter);
time = zeros(n_s, iter);
l_time = zeros(n_s, 1);
for ps = 1 : n_s
    s = s_para(ps)/k; % s = e

    t_start = tic;
    [ii, jj, ss] = find(S);
    [mm, nn] = size(S);
    s_mean = mean(ss);
    para_t = - full(s_mean) ./ log(s);
    W = sparse(ii, jj, exp(-ss ./ para_t), mm, nn);
    W(isnan(W)) = 0; W(isinf(W)) = 0;
    w_time = toc(t_start);
    
    % construct edges
    t_start = tic;
    [idx1, idx2] = find(W~=0);
    edges = [idx1 idx2 W(idx1+(idx2-1).*n)];
    e_time = toc(t_start);
    l_time(ps) = e_time + w_time;

    for t = 1 : iter 
        tic;
        label_ind = find(label(:,t));
        unlabel_ind = find(~label(:,t));
        % construct Y    
        Y = zeros(n, 1)-1;
        Y(label_ind) = Y_train(label_ind) - 1;
        % compute F
        F = mtc_matlab(full(edges), n, Y, n_class, 0, 1);
        F = F + 1;
        time(ps, t) = toc;
        % compute accuracy       
        e = mean(double(F(unlabel_ind) ~= Y_train(unlabel_ind)));    
        errs(ps, t) = e;
        % verbose
        fprintf('run_MTC: s = %e, t = %d, accuracy = % f\n', s_para(ps), t, 100*(1-e));
    end
end
err = mean(errs, 2);
[err, best_id] = min(err);
best_s = s_para(best_id);
v = std(errs, [], 2);
v = v(best_id);
time = mean(time(:));
l_time = mean(l_time(:));

end