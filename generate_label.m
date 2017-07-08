function label = generate_label(Y_train, para)
% randomly generate label indexes for semi-supervised learning

%%
fprintf('***randomly generate labels for semi-supervised learning***\n');
iter = para.iter;
type = para.type;
p = para.p;

%%
class = unique(Y_train);
n = numel(Y_train);
label = cell(numel(p), 1);
for i = 1 : numel(p)
    tmp_label = false(n, iter);
    switch type
        case 'skew'
            for t = 1 : iter
                label_ind = randsample(n, p(i));
                tmp_label(label_ind, t) = true;
            end
        case 'equal'
            for t = 1 : iter
                for j = 1 : numel(class)
                    if p(i) < 1
                        class_ind = find(Y_train == class(j));
                        np = floor(numel(class_ind) * p(i));
                        np = max(1, np);
                        label_ind = randsample(class_ind, np);
                    else
                        class_ind = find(Y_train == class(j));
                        if p(i) >= numel(class_ind)
                            np = floor(numel(class_ind) * 0.8);
                            np = max(1,np);
                        else
                            np = p(i);
                        end
                        label_ind = randsample(class_ind, np);
                    end
                    tmp_label(label_ind, t) = true;
                end
            end
        otherwise
    end
    label{i} = tmp_label;
end
   