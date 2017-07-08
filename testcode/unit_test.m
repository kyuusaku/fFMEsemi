%% unit test

%% test generate label
gnd = [1;2;3;1;2;3;1;2;3;1;1;2;3;2;3];

para.iter = 2;
para.type = 'equal';

para.p = 1;
label = generate_label(gnd, para);
label = label{1};
if (para.iter ~= size(label,2))
    error('iter wrong');
end
if (size(gnd,1) ~= size(label,1))
    error('size wrong');
end
for i = 1:para.iter
    if (unique(gnd) ~= unique(gnd(label(:,i))))
        error('sample class wrong');
    end
    tmp = hist(gnd(label(:,i)), unique(gnd));
    for j = 1:numel(tmp)
        if (tmp(j) ~= para.p)
            error('sample p wrong');
        end
    end
end

para.p = 2;
label = generate_label(gnd, para);
label = label{1};
if (para.iter ~= size(label,2))
    error('iter wrong');
end
if (size(gnd,1) ~= size(label,1))
    error('size wrong');
end
for i = 1:para.iter
    if (unique(gnd) ~= unique(gnd(label(:,i))))
        error('sample class wrong');
    end
    tmp = hist(gnd(label(:,i)), unique(gnd));
    for j = 1:numel(tmp)
        if (tmp(j) ~= para.p)
            error('sample p wrong');
        end
    end
end

para.p = 3;
label = generate_label(gnd, para);
label = label{1};
if (para.iter ~= size(label,2))
    error('iter wrong');
end
if (size(gnd,1) ~= size(label,1))
    error('size wrong');
end
for i = 1:para.iter
    if (unique(gnd) ~= unique(gnd(label(:,i))))
        error('sample class wrong');
    end
    tmp = hist(gnd(label(:,i)), unique(gnd));
    for j = 1:numel(tmp)
        if (tmp(j) ~= para.p)
            error('sample p wrong');
        end
    end
end