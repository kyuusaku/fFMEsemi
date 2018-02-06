%% unit test

%% test pca
points = mvnrnd([5,5], [3,0;0,0.1], 1000);
points = points';
scatter(points(1,:)',points(2,:)');
[U, M] = pca(points, 0)
R = [cosd(45), -sind(45); sind(45), cosd(45)]
points = R * points;


%% test generate label
gnd = [1;2;3;1;2;3;1;2;3;1;1;2;3;2;3];

para.iter = 2;
para.type = 'equal';

para.p = 1;
label = generate_label(gnd, para);
label = label{1};
if (para.iter ~= size(label,2))
    error('iter wrong');
end
if (size(gnd,1) ~= size(label,1))
    error('size wrong');
end
for i = 1:para.iter
    if (unique(gnd) ~= unique(gnd(label(:,i))))
        error('sample class wrong');
    end
    tmp = hist(gnd(label(:,i)), unique(gnd));
    for j = 1:numel(tmp)
        if (tmp(j) ~= para.p)
            error('sample p wrong');
        end
    end
end

para.p = 2;
label = generate_label(gnd, para);
label = label{1};
if (para.iter ~= size(label,2))
    error('iter wrong');
end
if (size(gnd,1) ~= size(label,1))
    error('size wrong');
end
for i = 1:para.iter
    if (unique(gnd) ~= unique(gnd(label(:,i))))
        error('sample class wrong');
    end
    tmp = hist(gnd(label(:,i)), unique(gnd));
    for j = 1:numel(tmp)
        if (tmp(j) ~= para.p)
            error('sample p wrong');
        end
    end
end

para.p = 3;
label = generate_label(gnd, para);
label = label{1};
if (para.iter ~= size(label,2))
    error('iter wrong');
end
if (size(gnd,1) ~= size(label,1))
    error('size wrong');
end
for i = 1:para.iter
    if (unique(gnd) ~= unique(gnd(label(:,i))))
        error('sample class wrong');
    end
    tmp = hist(gnd(label(:,i)), unique(gnd));
    for j = 1:numel(tmp)
        if (tmp(j) ~= para.p)
            error('sample p wrong');
        end
    end
end

%% FOR fFME
%% test XH
load('/home/lab-qiu.suo/devel/fFMEsemi/result/coil20/semi/pca.mat', 'X_train');
X = X_train;
[dim,n] = size(X);
Xc = bsxfun(@minus, X, mean(X,2));
one = ones(n,1);
H = eye(n) - 1/n.*one*one';
XH = X*H;
diff = Xc-XH;
assert(sum(abs(diff(:)) > 1e-5)==0, 'Xc and XH are not equal');

assert(one'*one==n, '11^T not equal to n');

%% test U
para.uu = 0; % uu must be zero
para.ul = 1;
Y = [1,0;
     0,0;
     0,1;
     0,0;];
n = size(Y,1);
assert(n==4, 'n error for Y');
u = para.uu .* ones(n,1);
assert(sum(u==zeros(n,1))==n, 'u error for init');
u(sum(Y,2) == 1) = para.ul;
assert(sum(u==[1;0;1;0])==n, 'u error for assign labeled data');
U = spdiags(u,0,n,n);
assert(sum(diag(full(U))==[1;0;1;0])==n, 'error when generate U');

para.mu = 1;
V_inv = spdiags((u + (para.mu + 1) .* ones(n,1)).^-1, 0, n, n);
assert(sum(diag(full(V_inv))==1./[3;2;3;2])==n, 'error when generate V_inv');
V = full(U) + (para.mu + 1) .* eye(n);
diff = V_inv - pinv(V);
assert(sum(abs(diff(:)) > 1e-5)==0, 'V_inv and pinv(V) are not equal');

