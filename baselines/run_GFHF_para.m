function result_GFHF = run_GFHF_para(Y_train, S, label, s, k)

%% default parameters
if ~exist('k', 'var')
    k = 10;
end
s_para = [1e-9;1e-7;1e-5;1e-3;1e-1];
if exist('s', 'var') && ~isempty(s)
    s_para = s;
end

%%
fprintf('******** Runing GFHF ***********\n');

class = unique(Y_train);

np = numel(label);
result_GFHF = cell(np, 1);
for i = 1 : np  
    [errs, err, v, time, l_time, best_s, best_id] = iner_run_GFHF(S, Y_train, class, label{i}, s_para, k);     
    result.accuracy = 100*(1-errs);
    result.best_train_accuracy = [100*(1-err), 100*v];
    result.best_s = best_s;
    result.best_id = best_id;
    result.average_time = time;
    result.l_time = l_time;
    result.p = sum(label{i});
    result.p = result.p(1);
    result_GFHF{i} = result;
    disp(result);
end    
fprintf('done.\n');
end

function [errs, err, v, time, l_time, best_s, best_id] = iner_run_GFHF(S, Y_train, class, label, s_para, k)
n_s = numel(s_para);
n = numel(Y_train);
n_class = numel(class);
iter = size(label, 2);
errs = zeros(n_s, iter);
time = zeros(n_s, iter);
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

    %% setting iteration
    for t = 1 : iter 
        tic;
        % construct Y
        label_ind = find(label(:,t));
        Y = zeros(n, n_class);
        for cc = 1 : n_class
            cc_ind = find(Y_train(label_ind) == cc);
            Y(label_ind(cc_ind), cc) = 1;
        end
        Y = sparse(Y);
        % compute F
        label_ind = find(label(:,t));
        unlabel_ind = find(~label(:,t));
        F = - L(unlabel_ind, unlabel_ind) \ ...
            (L(unlabel_ind, label_ind) * Y(label_ind, :)); 
        % normalization
        q = sum(Y(label_ind,:),1) + 1;
        F = F .* repmat(q ./ sum(F, 1), numel(unlabel_ind), 1); clear Y;
        time(ps, t) = toc;
        % compute accuracy
        [~, predictions] = max(F, [], 2); clear F;
        e = mean(double(predictions ...
                    ~= Y_train(unlabel_ind)));
        errs(ps, t) = e;
        % verbose
        fprintf('run_GFHF: s = %e, t = %d, accuracy = %f\n', s_para(ps), t, 100*(1-e));
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