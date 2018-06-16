function show_image_dataset(dataset)

data_path = 'data';
if strcmp(dataset, 'usps')
    load(fullfile(data_path, 'USPS', 'USPS.mat'));
    fea = fea';
    h = 16;
    w = 16;
    transpose = true;
elseif strcmp(dataset, 'coil100')
    load(fullfile(data_path, 'COIL100', 'COIL100.mat'));
    fea = fea';
    h = 32;
    w = 32;
    transpose = false;
elseif strcmp(dataset, 'norb')
    load(fullfile(data_path, 'norb_small_gray', 'norb_small_gray.mat'));
    fea = trainX;
    gnd = trainY;
    h = 96;
    w = 96;
    transpose = false;
elseif strcmp(dataset, 'usps-large')
    load(fullfile(data_path, 'Extended_USPS', 'Extended_USPS.mat'));
    fea = data';
    gnd = label;
    h = 26;
    w = 26;
    transpose = false;
elseif strcmp(dataset, 'mnist-large')
    load(fullfile(data_path, 'Extended_MNIST', 'Extended_MNIST.mat'));
    fea = data';
    gnd = label;
    h = 30;
    w = 30;
    transpose = false;
else
    return;
end

num_random_show = 5;
[~, nSample] = size(fea);
sample_inds = randsample(nSample, num_random_show);
samples_for_show = fea(:, sample_inds);
samples_for_show = reshape(samples_for_show, [h, w, num_random_show]);
figure;
for i = 1 : num_random_show
    subplot(1, num_random_show, i);
    if transpose
        imagesc(samples_for_show(:, :, i)');
    else
        imagesc(samples_for_show(:, :, i));
    end
    colormap(gray);
    title(num2str(gnd(sample_inds(i))));
end
