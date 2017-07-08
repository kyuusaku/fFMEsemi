function check_S()
% sample:
q = [1,1,1,1;0,1,1,1;0,0,1,1;0,0,0,1]
% 2 nn_graph:
a = [0,1,2,0;1,0,1,2;2,1,0,1;0,2,1,0]

s = full(constructS(q,2))
ss = full(knn_graph_max(q,2+1))

if sum(sum(s~=a))
    error('constructS is wrong');
end
if sum(sum(ss~=a))
    error('knn_graph_max is wrong');
end
