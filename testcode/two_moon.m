function [points, labels] = two_moon(n, std)

x = linspace(0, pi, n);
points = zeros(2*n, 2);
points(1:n,1) = cos(x) + normrnd(0,std,n,1)';
points(1:n,2) = sin(x) + normrnd(0,std,n,1)';
points(n+1:2*n,1) = 1 - cos(x) + normrnd(0,std,n,1)';
points(n+1:2*n,2) = 1 - sin(x) - 0.5 + normrnd(0,std,n,1)';

labels = ones(2*n, 1);
labels(n+1:2*n) = 2;