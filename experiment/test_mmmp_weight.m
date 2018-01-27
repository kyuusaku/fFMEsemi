%% test mmmp_weight

%% 
addpath('../fFME');
addpath('../mmlp');

%% input
E = [0,1,0,2,0,0,0,0,0;
     1,0,1,1,2,0,0,0,0;
     0,1,0,2,1,2,0,0,0;
     2,1,2,0,1,0,0,0,0;
     0,2,1,1,0,2,0,0,0;
     0,0,2,0,2,0,0,0,0;
     0,0,0,0,0,0,0,2,2;
     0,0,0,0,0,0,2,0,2;
     0,0,0,0,0,0,2,2,0];
E = sparse(E);

aIdx = [3;4;5;9]; 

k = 2;

%% answer
answer_D = [1,1,1,inf;
            1,1,1,inf;
            0,1,1,inf;
            1,0,1,inf;
            1,1,0,inf;
            2,2,2,inf;
            inf,inf,inf,2;
            inf,inf,inf,2;
            inf,inf,inf,0]

[val, pos] = sort(answer_D, 2);

val = val(:,1:k);
pos = pos(:,1:k);

val = val ./ (max(val, [], 2) * ones(1,k));
val = exp(-val);
val(isnan(val)) = 0;
val = val ./ (sum(val, 2) * ones(1,k));

n = size(E,1);
m = size(aIdx,1);
answer = sparse(reshape(repmat(1:n, k,1), 1,[]), reshape(pos', 1,[]), reshape(val', 1,[]), n, m);
answer = full(answer)

%% output
[Z, elapsed_time, D] = mmmp_weight(E, aIdx, k);
D
Z = full(Z)
