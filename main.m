%%
clear;
parpool(12);


%% unsupervised entropy
run_experiment_semi_with_dl_un('cifar10','_igan_feature_matching_unsupervised_entropy',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/cifar_feature_matching_unsupervised_entropy/seed1_seeddata1_count400/fea.mat',...
    400,'anchorNumber',1000)
%%
run_experiment_semi_with_dl('cifar10','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/cifar_feature_matching/seed1_seeddata1_count400/fea.mat',...
    400,'anchorNumber',4000)
run_experiment_semi_with_dl('cifar10','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/cifar_feature_matching/seed2_seeddata2_count400/fea.mat',...
    400,'anchorNumber',2000)
run_experiment_semi_with_dl('cifar10','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/cifar_feature_matching/seed3_seeddata3_count400/fea.mat',...
    400,'anchorNumber',2000)
run_experiment_semi_with_dl('cifar10','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/cifar_feature_matching/seed4_seeddata4_count400/fea.mat',...
    400,'anchorNumber',2000)
run_experiment_semi_with_dl('cifar10','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/cifar_feature_matching/seed5_seeddata5_count400/fea.mat',...
    400,'anchorNumber',2000)

%% unsupervised entropy
run_experiment_semi_with_dl_un('mnist','_igan_feature_matching_unsupervised_entropy',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/mnist_feature_matching_unsupervised_entropy/seed1_seeddata1_count10/fea.mat',...
    10,'anchorNumber',1000)
%% semi supervised
run_experiment_semi_with_dl('mnist','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/mnist_feature_matching/seed1_seeddata1_count10/fea.mat',...
    10,'anchorNumber',1000)
run_experiment_semi_with_dl('mnist','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/mnist_feature_matching/seed2_seeddata2_count10/fea.mat',...
    10,'anchorNumber',1000)
run_experiment_semi_with_dl('mnist','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/mnist_feature_matching/seed3_seeddata3_count10/fea.mat',...
    10,'anchorNumber',1000)
run_experiment_semi_with_dl('mnist','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/mnist_feature_matching/seed4_seeddata4_count10/fea.mat',...
    10,'anchorNumber',1000)
run_experiment_semi_with_dl('mnist','_igan_feature_matching',...
    'prepare/ssdl/igan_mnist_svhn_cifar10/output/mnist_feature_matching/seed5_seeddata5_count10/fea.mat',...
    10,'anchorNumber',1000)

%%
run_experiment_semi('usps',10,'anchorNumber',100,'runFME',true)
run_experiment_semi('usps',10,'anchorNumber',50)
run_experiment_semi('usps',10,'anchorNumber',200)
run_experiment_semi('usps',10,'anchorNumber',300)
run_experiment_semi('usps',10,'anchorNumber',400)
run_experiment_semi('usps',10,'anchorNumber',500)
run_experiment_semi('usps',10,'anchorNumber',600)
run_experiment_semi('usps',10,'anchorNumber',700)
run_experiment_semi('usps',10,'anchorNumber',800)
run_experiment_semi('usps',10,'anchorNumber',900)
run_experiment_semi('usps',10,'anchorNumber',1000)

%%
run_experiment_semi('coil100',10,'anchorNumber',100,'runFME',true)
run_experiment_semi('coil100',10,'anchorNumber',50)
run_experiment_semi('coil100',10,'anchorNumber',200)
%run_experiment_semi('coil100',10,'anchorNumber',300)
run_experiment_semi('coil100',10,'anchorNumber',400)
run_experiment_semi('coil100',10,'anchorNumber',600)
%run_experiment_semi('coil100',10,'anchorNumber',700)
run_experiment_semi('coil100',10,'anchorNumber',800)
%run_experiment_semi('coil100',10,'anchorNumber',900)
run_experiment_semi('coil100',10,'anchorNumber',1000)
run_experiment_semi('coil100',10,'anchorNumber',1200)
run_experiment_semi('coil100',10,'anchorNumber',1400)
run_experiment_semi('coil100',10,'anchorNumber',1600)
run_experiment_semi('coil100',10,'anchorNumber',1800)
run_experiment_semi('coil100',10,'anchorNumber',2000)
run_experiment_semi('coil100',10,'anchorNumber',2200)
run_experiment_semi('coil100',10,'anchorNumber',2400)
run_experiment_semi('coil100',10,'anchorNumber',2600)
run_experiment_semi('coil100',10,'anchorNumber',2800)
run_experiment_semi('coil100',10,'anchorNumber',3000)
run_experiment_semi('coil100',10,'anchorNumber',3200)
run_experiment_semi('coil100',10,'anchorNumber',3400)
run_experiment_semi('coil100',10,'anchorNumber',3600)
run_experiment_semi('coil100',10,'anchorNumber',3800)
run_experiment_semi('coil100',10,'anchorNumber',4000)

%%
run_experiment_semi('norb',1)
%%
run_experiment_semi('norb',5)
%%
run_experiment_semi('norb',8)
%%
run_experiment_semi('norb',10)
%%
run_experiment_semi('norb',15)
%%
run_experiment_semi('aloi',1)
%%
run_experiment_semi('aloi',5)
%%
run_experiment_semi('aloi',8)
%%
run_experiment_semi('aloi',10)
%%
run_experiment_semi('aloi',15)
%%
run_experiment_semi('rcv1',1)
%%
run_experiment_semi('rcv1',5)
%%
run_experiment_semi('rcv1',8)
%%
run_experiment_semi('rcv1',10)
%%
run_experiment_semi('rcv1',15)
%%
run_experiment_semi('rcv1',50)
%%
run_experiment_semi('rcv1',100)
%%
run_experiment_semi('mnist630k',1)
%%
run_experiment_semi('mnist630k',5)
%%
run_experiment_semi('mnist630k',8)
%%
run_experiment_semi('mnist630k',10)
%%
run_experiment_semi('mnist630k',15)
%%
run_experiment_semi('cifar10-rgb',1)
%%
run_experiment_semi('cifar10-rgb',5)
%%
run_experiment_semi('cifar10-rgb',8)
%%
run_experiment_semi('cifar10-rgb',10)
%%
run_experiment_semi('cifar10-rgb',15)
%%
run_experiment_semi('covtype',1)
%%
run_experiment_semi('covtype',5)
%%
run_experiment_semi('covtype',8)
%%
run_experiment_semi('covtype',10)
%%
run_experiment_semi('covtype',15)
%%
run_experiment_semi('covtype',50)
%%
run_experiment_semi('covtype',100)
%%
run_experiment_semi('usps-large',1)
%%
run_experiment_semi('usps-large',5)
%%
run_experiment_semi('usps-large',8)
%%
run_experiment_semi('usps-large',10)
%%
run_experiment_semi('usps-large',15)
%%
run_experiment_semi('mnist-large-imbalance',1)
%%
run_experiment_semi('mnist-large-imbalance',5)
%%
run_experiment_semi('mnist-large-imbalance',8)
%%
run_experiment_semi('mnist-large-imbalance',10)
%%
run_experiment_semi('mnist-large-imbalance',15)
%%
run_experiment_semi('usps-large-imbalance',1)
%%
run_experiment_semi('usps-large-imbalance',5)
%%
run_experiment_semi('usps-large-imbalance',8)
%%
run_experiment_semi('usps-large-imbalance',10)
%%
run_experiment_semi('usps-large-imbalance',15)