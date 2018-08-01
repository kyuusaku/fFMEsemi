%% unit test

%% test pca
points = mvnrnd([5,5], [3,0;0,0.1], 1000);
points = points';
scatter(points(1,:)',points(2,:)');
[U, M] = pca(points, 0)
R = [cosd(45), -sind(45); sind(45), cosd(45)]
points = R * points;

%% test runNN_para
n_samples_per_cluster = 200;
n_features = 2;
fea = randn(n_samples_per_cluster, n_features);
fea = [fea + repmat([-5, 0], n_samples_per_cluster, 1);...
       fea + repmat([5, 0], n_samples_per_cluster, 1)];
gnd = [ones(n_samples_per_cluster, 1); 
     2*ones(n_samples_per_cluster, 1)];
fea=fea';
gscatter(fea(1,:)', fea(2,:)', gnd);
split = choose_each_class(gnd, 0.5, 1);
X_train = fea(:, split); Y_train = gnd(split);
X_test = fea(:, ~split); Y_test = gnd(~split);
para.p = [1];
para.iter = 20;
para.type = 'equal';
label = generate_label(Y_train, para);
result_NN_para = run_NN_para(X_train, Y_train, X_test, Y_test, label);
result_NN_para{1}.accuracy

%% test imbalanced_sample
gnd = [1;1;1;2;2;2;3;3;3;4;4;4];
fea = gnd';
class = unique(gnd)
ratio = [0.5,0.8,1,0.6]
[Fea, Gnd] = imbalanced_sample(fea, gnd, class, ratio);
assert(numel(Gnd)==7);
assert(sum(Gnd==1)==1);
assert(sum(Gnd==2)==2);
assert(sum(Gnd==3)==3);
assert(sum(Gnd==4)==1);
assert(isequal(Fea,[1,2,2,3,3,3,4]));

%% test choose_each_class
gnd = [1;1;2;2;2;3;3;3;3;4;4;4;4;4];
split = choose_each_class(gnd, 0.8, 1);
assert(sum(split)==10);
gnd_split = gnd(split);
gnd_split_ = gnd(~split);
assert(isequal(gnd_split_, [1;2;3;4]));
assert(sum(gnd_split==1)==1);
assert(sum(gnd_split==2)==2);
assert(sum(gnd_split==3)==3);
assert(sum(gnd_split==4)==4);
split = choose_each_class(gnd, 0.5, 1);
assert(sum(split)==6);
gnd_split = gnd(split);
assert(sum(gnd_split==1)==1);
assert(sum(gnd_split==2)==1);
assert(sum(gnd_split==3)==2);
assert(sum(gnd_split==4)==2);
split = choose_each_class(gnd, 1, 1);
assert(sum(split)==4);
gnd_split = gnd(split);
assert(sum(gnd_split==1)==1);
assert(sum(gnd_split==2)==1);
assert(sum(gnd_split==3)==1);
assert(sum(gnd_split==4)==1);

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

%% test ridge regression
f = 2; c = 4; l = 10;
W = rand(f, c);
b = rand(c, 1);
Xl = rand(f, l);
Y = rand(l, c);
o = ones(l, 1);
sum1 = 0;
for i = 1:l
    tmp = W'*Xl(:,i) + b - Y(i,:)';
    tmp = tmp .* tmp;
    sum1 = sum1 + sum(tmp(:));
end
sum1 = sum1 / l
sum2 = trace(Xl'*W*W'*Xl+2*Xl'*W*b*o'-2*Xl'*W*Y'+o*b'*b*o'-2*o*b'*Y'+Y*Y') / l
    