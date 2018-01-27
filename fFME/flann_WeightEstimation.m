function [Z, elapsed_time] = flann_WeightEstimation(Data, Anchor, s, verbose)
% Z(nXm): output anchor-to-data regression weight matrix 

if ~exist('verbose', 'var') || isempty(verbose)
	verbose = false;
end

[d, m] = size(Anchor);
n = size(Data,2);

if verbose
	disp(['### Anchor graph construction with ' num2str(n) ' data points and ' num2str(m) ' anchor points ###']);
end

t_start = tic;

% Find s nearest anchor points
tic;
[kdtree, search_params] = flann_build_index(Anchor, struct('algorithm','kdtree', 'trees',8, 'checks',128));
search_params.cores = 0;

if verbose
	disp(['  kd-tree constructed...' num2str(toc)]);
end

tic;
[pos, val] = flann_search(kdtree, Data, s, search_params);
val = val.^2';
pos = pos';
flann_free_index(kdtree);

if verbose
	disp(['  ' num2str(s) '-nearest anchor points found...' num2str(toc)]);
end

% Compute the l1 normalized inter-layer weights   
sigma = mean(val(:,s).^0.5);
val = exp(-val / (1/1*sigma^2));
val = repmat(sum(val, 2).^-1, 1,s) .* val; 
    
Z = sparse(reshape(repmat(1:n, s,1), 1,[]), reshape(pos', 1,[]), reshape(val', 1,[]), n, m);

% Measure time
elapsed_time = toc(t_start);

if verbose
	disp([' Elapsed time = ' num2str(elapsed_time)]);
end
