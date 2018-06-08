function draw_anchors(dataset)

%%
if strcmp(dataset, 'usps')
    num_anchors = [50, 100 : 100 : 1000];
    FME_anchor = 100;
end
%%
if strcmp(dataset, 'coil100')
    num_anchors = 200 : 200 : 4000;
    FME_anchor = 100;
end

%%
save_path = ['result/' dataset '/'];
%%
load([save_path 'semi-a' num2str(FME_anchor) '/record-10/result_FME1_1_para_best.mat']); 
FME_train = result_FME1_1_para_best{1,1}.best_train_accuracy;
FME_test = result_FME1_1_para_best{1,1}.best_test_accuracy;

fFME_train = zeros(numel(num_anchors), 2);
fFME_test = zeros(numel(num_anchors), 2);
aFME_train = zeros(numel(num_anchors), 2);
aFME_test = zeros(numel(num_anchors), 2);
for i = 1:numel(num_anchors)
    record_path = [save_path 'semi-a' num2str(num_anchors(i)) '/record-10/'];
    load([record_path 'result_fastFME1_1e9_para_best2.mat']);
    fFME_train(i,:) = result_fastFME1_1e9_para_best{1,1}.best_train_accuracy;
    fFME_test(i,:) = result_fastFME1_1e9_para_best{1,1}.best_test_accuracy;
    load([record_path 'result_aFME_1e9_para_best.mat']);
    aFME_train(i,:) = result_aFME_1e9_para_best{1,1}.best_train_accuracy;
    aFME_test(i,:) = result_aFME_1e9_para_best{1,1}.best_test_accuracy;
end

save([save_path 'data.mat'],'FME_train','FME_test','fFME_train','fFME_test','aFME_train','aFME_test');
%%
load([save_path 'data.mat']);

h1=figure(1);
m_size = 15;
f_size = 21;

plot(num_anchors, repmat(FME_train(1),numel(num_anchors),1), '-k', 'LineWidth', 2); hold on;
plot(num_anchors, repmat(FME_test(1),numel(num_anchors),1), '--k', 'LineWidth', 2); hold on;
plot(num_anchors, fFME_train(:,1), '-b+', 'MarkerSize', m_size, 'LineWidth', 2); hold on;
plot(num_anchors, fFME_test(:,1), '--b+', 'MarkerSize', m_size, 'LineWidth', 2); hold on;
plot(num_anchors, aFME_train(:,1), '-m^', 'MarkerSize', m_size, 'LineWidth', 2); hold on;
plot(num_anchors, aFME_test(:,1), '--m^', 'MarkerSize', m_size, 'LineWidth', 2); hold on;

legend('FME(unlabel)','FME(test)','f-FME(unlabel)','f-FME(test)','r-FME(unlabel)','r-FME(test)',...
    'Location','SouthEast');
set(gca,'YLim',[0,100]);
set(gca,'XLim',[min(num_anchors),max(num_anchors)]);
xlabel('Number of anchors','FontSize',f_size);
ylabel('Unlabel & Test accuracy','FontSize',f_size);
set(gca,'FontSize',f_size);
print(h1,'-dpng',fullfile(save_path, [dataset '-anchors.png']));