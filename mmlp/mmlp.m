function [y_pred, error, fp, fn, elapsed_time, num_iter, num_prop, iters] = mmlp(E, X, y, label_index, pairs_intra, pairs_extra, verbose)

if ~exist('verbose', 'var') || isempty(verbose)
	verbose = false;
end

N = length(y);

if verbose
	disp(['### Minimax label propagation with ' num2str(N) ' data points and ' num2str(max(y)) ' classes ###']);
end

t_start = tic;

%% Initialization
if N < 10000
	%% Initializing predicted labels by 1-NN classification
	% Good for disconnection from labeled nodes, taking insignificant time on small-scale data
	X_labeled = X(:,label_index);
	[index, search_params] = flann_build_index(X_labeled, struct('algorithm','kdtree', 'trees',8, 'checks',128));
	[gIdx, dist] = flann_search(index, X, 1, search_params);
	y_pred = y(label_index(gIdx'));
	y_pred(label_index) = y(label_index);
	d = dist' + 100;
	d(label_index) = 0;
else
	y_pred = zeros(length(y), 1);
	y_pred(label_index) = y(label_index);
	d = inf(length(y), 1);
	d(label_index) = 0;
end

%% Main algorithm
[num_iter, size_Q, iters] = mmlp_core(label_index, E, y_pred, d, 0.9999);

%% Measure time
elapsed_time = toc(t_start);

if verbose
    disp([' Elapsed time = ' num2str(elapsed_time)]);
end

%% Measure error
error = 100 * sum(y_pred ~= y) / (N - length(label_index));

%% Measure FP and FN
if ~exist('pairs_intra', 'var') || isempty(pairs_intra)
	fn = [];
else
	fn = 100 * mean(y_pred(pairs_intra(:,1)) ~= y_pred(pairs_intra(:,2)));
end

if ~exist('pairs_extra', 'var') || isempty(pairs_extra)
	fp = [];
else
	fp = 100 * mean(y_pred(pairs_extra(:,1)) == y_pred(pairs_extra(:,2)));
end

%% Measure the number of propagation operations (= computational cost)
num_prop = sum(size_Q);

%% Measure the histogram of iteration counts for convergence
iters = hist(iters, 1:30);
