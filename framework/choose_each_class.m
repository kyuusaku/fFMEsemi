function split = choose_each_class(gnd, p, t)
% black test
split = false(numel(gnd), t);
class = unique(gnd);
for t = 1 : t
    for j = 1 : numel(class)
        if p < 1
            class_ind = find(gnd == class(j));
            np = floor(numel(class_ind) * p);
            label_ind = randsample(class_ind, np);
        else
            label_ind = randsample(find(gnd == class(j)), p);
        end
        split(label_ind, t) = true;
    end
end

end