function [X_train, Y_train, X_test, Y_test] = load_dataset(dataset,para)

data_path = fullfile('data', para.dataset);
if ~exist(data_path, 'dir')
    error('no dataset exists');
end

if strcmp(dataset, 'norb')
    % load original data
    load(fullfile(data_path, strcat(para.dataset, '.mat')));
    % preprocess
    X_train = trainX; Y_train = trainY + 1;
    X_test = testX; Y_test = testY + 1;
end