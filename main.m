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
run_experiment_semi('usps',10,'anchorNumber',1000,'runFME',true)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',50)
run_experiment_semi('usps',10,'anchorNumber',50)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',100)
run_experiment_semi('usps',10,'anchorNumber',100)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',200)
run_experiment_semi('usps',10,'anchorNumber',200)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',300)
run_experiment_semi('usps',10,'anchorNumber',300)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',400)
run_experiment_semi('usps',10,'anchorNumber',400)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',500)
run_experiment_semi('usps',10,'anchorNumber',500)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',600)
run_experiment_semi('usps',10,'anchorNumber',600)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',700)
run_experiment_semi('usps',10,'anchorNumber',700)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',800)
run_experiment_semi('usps',10,'anchorNumber',800)
copy_files_for_different_anchor_same_label('usps',10,'anchorNumber',900)
run_experiment_semi('usps',10,'anchorNumber',900)
draw_anchors('usps');

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
run_experiment_semi('usps-large-imbalance',5,'anchorNumber',2000,'classMassNormalization',false)
run_experiment_semi('usps-large-imbalance',8,'anchorNumber',2000,'classMassNormalization',false)
run_experiment_semi('usps-large-imbalance',10,'anchorNumber',2000,'classMassNormalization',false)

%%
run_experiment_semi('rcv1',30,'classMassNormalization',false)
run_experiment_semi('rcv1',50,'classMassNormalization',false)
run_experiment_semi('rcv1',70,'classMassNormalization',false)

%%
run_experiment_semi('covtype',30,'classMassNormalization',false)
run_experiment_semi('covtype',50,'classMassNormalization',false)
run_experiment_semi('covtype',70,'classMassNormalization',false)

