
fea_file = 'igan_mnist_svhn_cifar10/output/mnist_feature_matching/seed1_seeddata1_count10/fea_tsne.mat';
load(fea_file);
gscatter(testx(:,1), testx(:,2), testy');
gscatter(trainx(:,1), trainx(:,2), trainy');