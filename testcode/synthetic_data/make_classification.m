function [X, y] = make_classification(n_samples, n_features, n_classes)
%
% n_samples: The total number of samples
% n_features: The total number of features. All are informative features. 
% n_classes: The number of classes of the classification problem. One cluster per class.
% 
% X: array of shape [n_samples, n_features]
% y: array of shape [n_samples]
%

n_clusters = n_classes;
n_samples_per_cluster = n_samples / n_clusters;

X = zeros(n_samples, n_features);
y = zeros(n_samples, 1);

centroids = 2*n_clusters*lhsdesign(n_clusters, n_features);

for i = 1:n_clusters
	X((i-1)*n_samples_per_cluster+1:i*n_samples_per_cluster,:) = ...
        randn(n_samples_per_cluster, n_features) + repmat(centroids(i,:), n_samples_per_cluster, 1);
    y((i-1)*n_samples_per_cluster+1:i*n_samples_per_cluster) = i;
end
