function [E, elapsed_time, mean_10th] = knn_graph_max(X, k, checks, verbose)

if ~exist('k', 'var') || isempty(k)
	k = 20;
end
if ~exist('checks', 'var') || isempty(checks)
	checks = 128;
end
if ~exist('verbose', 'var') || isempty(verbose)
	verbose = false;
end

N = size(X,2);

if verbose
	disp(['### ' num2str(k) '-NN graph construction for ' num2str(N) ' data points ###' ]);
end

tstart = tic;

%% Find k-NNs
tic;
[kdtree, search_params] = flann_build_index(X, struct('algorithm','kdtree', 'trees',8, 'checks',checks));
search_params.cores = 0;

if verbose
    disp(['  kd-tree constructed...' num2str(toc)]);
end

tic;
[index, dist] = flann_search(kdtree, X, k, search_params);
index = index';
dist = dist';
flann_free_index(kdtree);

if verbose
    disp(['  Nearest neighbors found...' num2str(toc)]);
end

%% Graph construction
tic;
dist(dist <= 0) = eps;
E = sparse(repmat((1:N)', k,1), index(:), dist(:), N, N);
E(1:N+1:end) = 0;
E = max(E, E');

if verbose
    disp(['  Graph constructed...' num2str(toc)]);
end

%% Compute the mean of 10-th smallest distances
mean_10th = mean(dist(:, min(k,10)));

%% Measure Time
elapsed_time = toc(tstart);

if verbose
    disp([' Elapsed time = ' num2str(elapsed_time)]);
end
