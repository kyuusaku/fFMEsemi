function [points, labels] = cross_gaussian(n, d)

points = zeros(2*n, 2);
points(1:n,:) = mvnrnd([4,8], [3,0;0,0.1], n);
points(n+1:2*n,:) = mvnrnd([d, 4+d], [0.1,0;0,3], n);
labels = ones(2*n, 1);
labels(n+1:2*n) = 2;