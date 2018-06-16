%%
clear;
parpool;
addpath(genpath('testcode'));

%% Running time on synthetic data
run_time('samples', [10000, 20000, 40000, 80000, 160000], 10);
run_time('features', 10000, [256, 512, 1024, 2048, 4096]);

%% Performance on synthetic data
run_experiment_semi('two_moon',2,'anchorNumber',100);
run_experiment_semi('halfkernel',2,'anchorNumber',100);
run_experiment_semi('pinwheel',2,'anchorNumber',100);

%% Performance on small dataset with different number of anchors
%%
run_experiment_semi('usps',10,'anchorNumber',50)
run_experiment_semi('usps',10,'anchorNumber',100,'runFME',true)
run_experiment_semi('usps',10,'anchorNumber',200)
run_experiment_semi('usps',10,'anchorNumber',300)
run_experiment_semi('usps',10,'anchorNumber',400)
run_experiment_semi('usps',10,'anchorNumber',500)
run_experiment_semi('usps',10,'anchorNumber',600)
run_experiment_semi('usps',10,'anchorNumber',700)
run_experiment_semi('usps',10,'anchorNumber',800)
run_experiment_semi('usps',10,'anchorNumber',900)
run_experiment_semi('usps',10,'anchorNumber',1000)
draw_anchors('usps');

%%
run_experiment_semi('coil100',10,'anchorNumber',100,'runFME',true)
run_experiment_semi('coil100',10,'anchorNumber',200)
run_experiment_semi('coil100',10,'anchorNumber',400)
run_experiment_semi('coil100',10,'anchorNumber',600)
run_experiment_semi('coil100',10,'anchorNumber',800)
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
draw_anchors('coil100');

%% Performance on real-world large scale dataset
%%
run_experiment_semi('norb',5)
run_experiment_semi('norb',8)
run_experiment_semi('norb',10)

%%
run_experiment_semi('mnist-large-imbalance',5,'anchorNumber',2000)
run_experiment_semi('mnist-large-imbalance',8,'anchorNumber',2000)
run_experiment_semi('mnist-large-imbalance',10,'anchorNumber',2000)

%%
run_experiment_semi('usps-large-imbalance',5,'anchorNumber',2000)
run_experiment_semi('usps-large-imbalance',8,'anchorNumber',2000)
run_experiment_semi('usps-large-imbalance',10,'anchorNumber',2000)

%%
run_experiment_semi('rcv1',10,'classMassNormalization',false)
run_experiment_semi('rcv1',50,'classMassNormalization',false)
run_experiment_semi('rcv1',100,'classMassNormalization',false)

%%
run_experiment_semi('rcv1',10,'anchorNumber',2000)
run_experiment_semi('rcv1',50,'anchorNumber',2000)
run_experiment_semi('rcv1',100,'anchorNumber',2000)

%%
run_experiment_semi('covtype',10,'classMassNormalization',false)
run_experiment_semi('covtype',50,'classMassNormalization',false)
run_experiment_semi('covtype',100,'classMassNormalization',false)

%%
run_experiment_semi('covtype',10,'anchorNumber',2000)
run_experiment_semi('covtype',50,'anchorNumber',2000)
run_experiment_semi('covtype',100,'anchorNumber',2000)