dataset = single(rand(128,10000));
testset = single(rand(128,1000));
params.algorithm='kdtree';
params.trees=2;
params.checks=16;
index = flann_build_index(dataset,params);
flann_save_index(index,'flann_kdtree.idx')
