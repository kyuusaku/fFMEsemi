function [Fea, Gnd] = imbalanced_sample(fea, gnd, class, ratio)
% have tested

sample = false(numel(gnd), 1);
for i = 1 : numel(class)
    ind = find(gnd == class(i));
    n = floor(numel(ind) * ratio(i));
    sample(randsample(ind, n)) = true;
end

Fea = fea(:, sample);
Gnd = gnd(sample);