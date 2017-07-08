function unit_test()
test_Y_U();

function test_Y_U()
label = false(3,1); label(1:2) = true
Y_train = [1;3;2]
a_Y = zeros(3,3); a_Y(1,1) = 1; a_Y(2,3) = 1
a_U = eye(3); a_U(3,3) = 0

%~~
n = numel(Y_train);
class = unique(Y_train);
n_class = numel(class);
label_ind = find(label);
Y = zeros(n, n_class);
for cc = 1 : n_class
    cc_ind = find(Y_train(label_ind) == cc);
    Y(label_ind(cc_ind),cc) = 1;
end
Y
%~~

%~~
para.uu = 0; para.ul = 1;
u = para.uu .* ones(n,1);
u(sum(Y,2) == 1) = para.ul;
U = spdiags(u,0,n,n);
U = full(U)
%~~

if sum(sum(abs(a_Y - Y))) ~= 0
    error('Y is wrong');
end
if sum(sum(abs(a_U - U))) ~= 0
    error('U is wrong');
end

para.mu = 0.1;
a_V = diag([2.1,2.1,1.1])
a_V_inv = diag([2.1,2.1,1.1].^-1)
%~~
V_inv = spdiags((u + (para.mu + 1) .* ones(n,1)).^-1, 0, n, n);
V_inv = full(V_inv)
%~~
if sum(sum(abs(a_V_inv - V_inv))) ~= 0
    error('V is wrong');
end
V_V_inv = a_V*V_inv
if sum(sum(abs(eye(n) - V_V_inv))) ~= 0
    error('V is wrong');
end