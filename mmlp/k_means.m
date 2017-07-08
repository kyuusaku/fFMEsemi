function [gIdx, C, elapsed_time] = k_means(X, k, max_iter, verbose)

if ~exist('max_iter', 'var') || isempty(max_iter)
	max_iter = 10;
end
if ~exist('verbose', 'var') || isempty(verbose)
	verbose = false;
end

N = size(X,2);

%% Check whether the second input argument is a set of k centroids
if ~isscalar(k)
	C = k;
	k = size(C, 2);
else
	rp = randperm(N)';
	rp = sort(rp(1:k));
	C = X(:,rp);
end

%% Allocating variables
g0 = ones(N, 1);
z = zeros(N, 1);
gIdx = zeros(N, 1);
num_iter = 1;

if verbose
	disp(['### Clustering ' num2str(N) ' data into ' num2str(k) ' centroids ###']);
end

t_start_elapsed = tic;
t_start = tic;

%% Main loop converge if previous partition is the same as the current one
while num_iter <= max_iter && any(g0 ~= gIdx)
	if verbose
		disp(['Iter ' num2str(num_iter) ': Time=' num2str(toc(t_start)) ', Diff=' num2str(sum(g0 ~= gIdx)) ', Err=' num2str(sum(z))])
		t_start = tic;
	end

	g0 = gIdx;

	%% Partition data to closest centroids
	tic;
	[index, search_params] = flann_build_index(C, struct('algorithm','kdtree', 'trees',8, 'checks',128));
    search_params.cores = 0;

	if verbose
		disp(['  kd-tree constructed...' num2str(toc)]);
        disp(search_params);
	end

	tic;
	[gIdx, z] = flann_search(index, X, 1, search_params);
	gIdx = gIdx';
	z = z';

	if verbose
		disp(['  Nearest centroids found...' num2str(toc)]);
	end

	flann_free_index(index);

	%% Update centroids from the new partition
	tic;
	for i = 1:k
		idx = find(gIdx == i);
		if isempty(idx)
			[~, maxidx] = max(z);
			C(:,i) = X(:,maxidx);
			gIdx(maxidx) = i;
			z(maxidx) = 0;
		else
			C(:,i) = mean(X(:,idx), 2);
		end
	end

	if verbose
		disp(['  Centroids updated...' num2str(toc) ' ' num2str(i)]);
		tic;
	end

	num_iter = num_iter+1;
end

%% Measure time
elapsed_time = toc(t_start_elapsed);

if verbose
	disp([' Elapsed time = ' num2str(elapsed_time)]);
end
