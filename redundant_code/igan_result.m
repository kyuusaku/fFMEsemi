%%

%% MNIST
save_path = 'result/mnist_igan_feature_matching/';
count = 10;
IGAN = [0.0190, 0.0087, 0.0096, 0.0084, 0.0097];
%% CIFAR-10
save_path = 'result/cifar10_igan_feature_matching/';
count = 400;
IGAN = [0.1701, 0.1721, 0.1694, 0.1629, 0.1653];
%%
IGAN = 100 - IGAN * 100;
%%
fFME = zeros(1, 5);
efFME = zeros(1, 5);
aFME = zeros(1, 5);
LapRLS = zeros(1, 5);
for i = 1 : 5
    record_path = [save_path 'semi-a1000-seed' num2str(i) '/record-' num2str(count)];
    load([record_path '/result_fastFME1_1e9_para_best2.mat']);
    fFME(i) = result_fastFME1_1e9_para_best{1}.best_test_accuracy(1);
    load([record_path '/result_efFME1_1e9_para_best2.mat']);
    efFME(i) = result_efFME1_1e9_para_best{1}.best_test_accuracy(1);
    load([record_path '/result_aFME_1e9_para_best.mat']);
    aFME(i) = result_aFME_1e9_para_best{1}.best_test_accuracy(1);
    load([record_path '/result_LapRLS2_para_best.mat']);
    LapRLS(i) = result_LapRLS2_para_best{1}.best_test_accuracy(1);
end
%%
all = [IGAN; fFME; efFME; aFME; LapRLS];
