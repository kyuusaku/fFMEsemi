function [anchors, anchors_time] = hanchors(X_train, num_anchors)
% num_anchors: a vector [m1, m2, ..., mh]

n_hierarchy = numel(num_anchors);
anchors = cell(n_hierarchy, 1);
anchors_time = zeros(n_hierarchy, 1);

[~, anchors{1}, anchors_time(1)] = k_means(X_train, num_anchors(1));
for i = 2 : n_hierarchy
    [~, anchors{i}, anchors_time(i)] = k_means(anchors{i-1}, num_anchors(i));
end