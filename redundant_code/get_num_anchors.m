function num_anchors = get_num_anchors(dataset)

%default
base = 1000;
h = 3;

if strcmp(dataset, 'norb')
    n = 48600;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'aloi')
    n = 86400;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'rcv1')
    n = 414831;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'mnist630k')
    n = 503996;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'cifar10-rgb')
    n = 50000;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'covtype')
    n = 464810;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'usps-large')
    n = 705768;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'usps-large-imbalance')
    n = 485220;
    num_anchors = get_num(n,base,h);
end
if strcmp(dataset, 'mnist-large-imbalance')
    n = 277200;
    num_anchors = get_num(n,base,h);
end

end

function nums = get_num(n,base,h)
a = (n/base)^(1/h);
nums = zeros(1,h);
nums(h) = base;
for i = h-1 : -1 : 1
    nums(i) = ceil(a * nums(i+1));
end
end