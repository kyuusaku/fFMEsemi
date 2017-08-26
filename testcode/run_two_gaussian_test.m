%% run two gaussian test
close all;
d = 1;
for std = 0.01:0.005:0.07
%     [fea, gnd] = two_gaussian(5000, d, std);
%     fea=fea';
%     figure;
%     gscatter(fea(1,:)', fea(2,:)', gnd);
    two_gaussian_test(d, std);
end

%% show
d = 1;
std = 0.01:0.005:0.07;

agr_train = zeros(numel(std),1);
eagr_train = zeros(numel(std),1);
mmlp_train = zeros(numel(std),1);
mtc_train = zeros(numel(std),1);
nn_train = zeros(numel(std),1);
laprls_train = zeros(numel(std),1);
ffme_train = zeros(numel(std),1);
effme_train = zeros(numel(std),1);
afme_train = zeros(numel(std),1);
nn_test = zeros(numel(std),1);
laprls_test = zeros(numel(std),1);
ffme_test = zeros(numel(std),1);
effme_test = zeros(numel(std),1);
afme_test = zeros(numel(std),1);

save_path = 'result/test/semi';
for i = 1:numel(std)
    record_path = fullfile(save_path, ['record_2017-gaussian-' num2str(d) '-' num2str(std(i))]);
    
    ffme_data_1_1e9_para_best = fullfile(record_path, 'result_fastFME1_1e9_para_best2.mat');
    load(ffme_data_1_1e9_para_best);
    effme_data_1e9_para_best = fullfile(record_path, 'result_efFME_1e9_para_best.mat');
    load(effme_data_1e9_para_best);
    afme_data_1e9_para_best = fullfile(record_path, 'result_aFME_1e9_para_best.mat');
    load(afme_data_1e9_para_best);
    agr_data_para_best = fullfile(record_path, 'result_AGR_para_best.mat');
    load(agr_data_para_best);
    eagr_data_para_best = fullfile(record_path, 'result_EAGR_para_best.mat');
    load(eagr_data_para_best);
    mmlp_data_para = fullfile(record_path, 'result_MMLP_min_para.mat');
    load(mmlp_data_para);
    mmlp_data_para = fullfile(record_path, 'result_MMLP_max_para.mat');
    load(mmlp_data_para);
    if result_MMLP_min_para{1}.best_train_accuracy(1) >= result_MMLP_max_para{1}.best_train_accuracy(1)
        result_MMLP_para = result_MMLP_min_para;
    else
        result_MMLP_para = result_MMLP_max_para;
    end
    mtc_data_para = fullfile(record_path, 'result_MTC_para.mat');
    load(mtc_data_para);
    nn_data_para = fullfile(record_path, 'result_NN_para.mat');
    load(nn_data_para);
    laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');
    load(laprls_data2_para_best);
    
    agr_train(i) = result_AGR_para_best{1}.best_train_accuracy(1);
    eagr_train(i) = result_EAGR_para_best{1}.best_train_accuracy(1);
    mmlp_train(i) = result_MMLP_para{1}.best_train_accuracy(1);
    mtc_train(i) = result_MTC_para{1}.best_train_accuracy(1);
    nn_train(i) = result_NN_para{1}.best_train_accuracy(1);
    laprls_train(i) = result_LapRLS2_para_best{1}.best_train_accuracy(1);
    ffme_train(i) = result_fastFME1_1e9_para_best{1}.best_train_accuracy(1);
    effme_train(i) = result_efFME_1e9_para_best{1}.best_train_accuracy(1);
    afme_train(i) = result_aFME_1e9_para_best{1}.best_train_accuracy(1);
    nn_test(i) = result_NN_para{1}.best_test_accuracy(1);
    laprls_test(i) = result_LapRLS2_para_best{1}.best_test_accuracy(1);
    ffme_test(i) = result_fastFME1_1e9_para_best{1}.best_test_accuracy(1);
    effme_test(i) = result_efFME_1e9_para_best{1}.best_test_accuracy(1);
    afme_test(i) = result_aFME_1e9_para_best{1}.best_test_accuracy(1);
end

h=figure(1);
m_size = 10;
f_size = 20;
subplot(1,2,1);
plot(std,agr_train, '-.c', 'MarkerSize',m_size);hold on;
plot(std,eagr_train, '--co', 'MarkerSize',m_size);hold on;
plot(std,mmlp_train, '--m<', 'MarkerSize',m_size);hold on;
plot(std,mtc_train, '--m>', 'MarkerSize',m_size);hold on;
plot(std,nn_train, '--k*', 'MarkerSize',m_size);hold on;
plot(std,laprls_train, '--bs', 'MarkerSize',m_size);hold on;
%plot(std,ffme_train, '-rd', 'LineWidth', 2, 'MarkerSize',m_size);hold on;
plot(std,effme_train, '-rd', 'LineWidth', 2, 'MarkerSize',m_size);hold on;
plot(std,afme_train, '-r^', 'LineWidth', 2, 'MarkerSize',m_size);hold on;
legend('AGR','EAGR','MMLP','MTC','NN','LapRLS/L','f-FME','a-FME',...
    'Location','SouthWest');
set(gca,'FontSize',f_size);
xlabel('variance of the distribution');
ylabel('Unlabel accuracy');
subplot(1,2,2);
plot(std,nn_test, '--k*', 'MarkerSize',m_size);hold on;
plot(std,laprls_test, '--bs', 'MarkerSize',m_size);hold on;
%plot(std,ffme_test, '-rd', 'LineWidth', 2, 'MarkerSize',m_size);hold on;
plot(std,effme_test, '-rd', 'LineWidth', 2, 'MarkerSize',m_size);hold on;
plot(std,afme_test, '-r^', 'LineWidth', 2, 'MarkerSize',m_size);hold on;
legend('NN','LapRLS/L','f-FME','a-FME',...
    'Location','SouthWest');
%set(gca, 'XTick', std);
%set(gca, 'XTickLabel', {'3,750','7,500','15,000','30,000','60,000'});
set(gca,'YLim',[75,100]);
set(gca,'FontSize',f_size);
%grid on;
xlabel('variance of the distribution');
ylabel('Test accuracy');
% print(h, '..\\..\\t.epsc', '-depsc');

%% 
%min(mnistk32graph, [], 1);
