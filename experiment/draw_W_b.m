function draw_W_b(W, b, fea, gnd)

gscatter(fea(1,:)', fea(2,:)', gnd);
hold on;
x1 = min(fea(1,:)) : max(fea(1,:));
c = numel(b);
x2 = cell(c,1);
for i = 1:c
    k = - W(1,i) / W(2,i);
    d = - b(i) / W(2,i);
    x2{i} = k*x1 + d;    
end
plot(x1, x2{1}, x1, x2{2}, x1, x2{3}, x1, x2{4});