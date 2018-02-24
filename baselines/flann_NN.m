function Y_test = flann_NN(X_train, Y_train, X_test)

[index, search_params] = flann_build_index(X_train, ...
    struct('algorithm', 'kdtree', 'trees', 8, 'checks', 128));
search_params.cores = 0;

gIdx = flann_search(index, X_test, 1, search_params);
flann_free_index(index);

Y_test = Y_train(gIdx);