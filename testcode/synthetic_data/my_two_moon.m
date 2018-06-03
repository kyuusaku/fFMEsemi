function [points, labels] = my_two_moon(n)

x = linspace(0, pi, n);
r = sqrt(rand(1, n));
points = zeros(2*n, 2);
points(1:n,1) = cos(x) .* (10 + r * 5);
points(1:n,2) = sin(x) .* (10 + r * 5);
x = linspace(0, pi, n);
points(n+1:2*n,1) = cos(x) .* (10 + r * 5) + 10;
points(n+1:2*n,2) = - sin(x) .* (10 + r * 5) + 5;

labels = ones(2*n, 1);
labels(n+1:2*n) = 2;