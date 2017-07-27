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
    % preprocess
    [U, M] = pca(X_train, para.pca_preserve);
    X_train = U'*bsxfun(@minus, X_train, M);
    X_test = U'*bsxfun(@minus, X_test, M);
end

if strcmp(dataset, 'rcv1')
    % load original data
    load(fullfile(data_path, strcat(para.dataset, '.mat')));
    fea = fea';
    % default split
    split = choose_each_class(gnd, 0.8, 1);
    % preprocess
    X_train = fea(:, split); Y_train = gnd(split);
    X_test = fea(:, ~split); Y_test = gnd(~split);
    clear fea gnd split;
    X_pca = X_train(:, randsample(1:numel(Y_train), 10000));
    X_pca = full(X_pca);
    [U, M] = pca(X_pca, para.pca_preserve); clear X_pca;
    X_train_tmp = zeros(para.pca_preserve, size(X_train, 2));
    for i = 1 : size(X_train, 2)
        X_train_tmp(:, i) = U' * (X_train(:,i) - M);
    end
    X_train = X_train_tmp; clear X_train_tmp;
    X_test_tmp = zeros(para.pca_preserve, size(X_test, 2));
    for i = 1 : size(X_test, 2)
        X_test_tmp(:, i) = U' * (X_test(:,i) - M);
    end
    X_test = X_test_tmp; clear X_test_tmp;
end

if strcmp(dataset, 'mnist630k')
    % load original data
    data = load(fullfile(data_path, strcat(para.dataset, '.mat')));
    fea = data.X; gnd = data.y; clear data;
    % default split
    split = choose_each_class(gnd, 0.8, 1);
    % preprocess
    X_train = fea(:, split); Y_train = gnd(split);
    X_test = fea(:, ~split); Y_test = gnd(~split);
    clear fea gnd split;
    [U, M] = pca(X_train, para.pca_preserve);
    X_train = U'*bsxfun(@minus, X_train, M);
    X_test = U'*bsxfun(@minus, X_test, M);
    clear U M;
end

if strcmp(dataset, 'cifar10-rgb')
    % load original data
    load(fullfile(data_path, strcat(para.dataset, '.mat')));
    % preprocess
    X_train = trainX'; Y_train = trainY;
    X_test = testX'; Y_test = testY;
    % preprocess
    [U, M] = pca(X_train, para.pca_preserve);
    X_train = U'*bsxfun(@minus, X_train, M);
    X_test = U'*bsxfun(@minus, X_test, M);
    clear U M;
end

if strcmp(dataset, 'covtype')
    % load original data
    load(fullfile(data_path, strcat(para.dataset, '.mat')));
    % default split
    split = choose_each_class(gnd, 0.8, 1);
    % preprocess
    X_train = fea(:, split); Y_train = gnd(split);
    X_test = fea(:, ~split); Y_test = gnd(~split);
    clear fea gnd split;
    [U, M] = pca(X_train, para.pca_preserve);
    X_train = U'*bsxfun(@minus, X_train, M);
    X_test = U'*bsxfun(@minus, X_test, M);
    clear U M;
end

if strcmp(dataset, 'usps-large')
    % load original data
    data = load(fullfile(data_path, strcat(para.dataset, '.mat')));
    fea = data.data'; gnd = data.label; clear data;
    % default split
    split = choose_each_class(gnd, 0.8, 1);
    % preprocess
    X_train = fea(:, split); Y_train = gnd(split);
    X_test = fea(:, ~split); Y_test = gnd(~split);
    clear fea gnd split;
    [U, M] = pca(X_train, para.pca_preserve);
    X_train = U'*bsxfun(@minus, X_train, M);
    X_test = U'*bsxfun(@minus, X_test, M);
    clear U M;
end