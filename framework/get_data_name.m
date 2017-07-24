function data_name = get_data_name(dataset)

if strcmp(dataset, 'norb')
    data_name = 'norb_small_gray';
end
if strcmp(dataset, 'rcv1')
    data_name = 'RCV1';
end
if strcmp(dataset, 'mnist630k')
    data_name = 'mnist630k';
end