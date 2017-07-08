function [points, labels] = two_gaussian(n, d, std)

cx = 0; cy = 0;
points = zeros(2*n, 2);
points(1:n,:) = mvnrnd([cx,cy], [std,0;0,std], n);
points(n+1:2*n,:) = mvnrnd([cx+d, cy], [std,0;0,std], n);
labels = ones(2*n, 1);
labels(n+1:2*n) = 2;