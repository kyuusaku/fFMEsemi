function data_name = get_data_name(dataset)

if strcmp(dataset, 'norb')
    data_name = 'norb_small_gray';
end
if strcmp(dataset, 'aloi')
    data_name = 'aloi';
end
if strcmp(dataset, 'rcv1')
    data_name = 'RCV1';
end
if strcmp(dataset, 'mnist630k')
    data_name = 'mnist630k';
end
if strcmp(dataset, 'cifar10-rgb')
    data_name = 'cifar10';
end
if strcmp(dataset, 'covtype')
    data_name = 'covtype';
end
if strcmp(dataset, 'coil100')
    data_name = 'COIL100';
end
if strcmp(dataset, 'usps')
    data_name = 'USPS';
end
if strcmp(dataset, 'usps-large')
    data_name = 'Extended_USPS';
end
if strcmp(dataset, 'usps-large-imbalance')
    data_name = 'Extended_USPS';
end
if strcmp(dataset, 'mnist-large')
    data_name = 'Extended_MNIST';
end
if strcmp(dataset, 'mnist-large-imbalance')
    data_name = 'Extended_MNIST';
end