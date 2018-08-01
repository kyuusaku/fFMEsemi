function show_with_variant_parameters(accuracy)
% default the last dimension is train or test 
num_of_dim = ndims(accuracy);
if num_of_dim == 4
    train_accuracy = accuracy(:,:,:,1);
    train_accuracy = mean(train_accuracy, 3);
    test_accuracy = accuracy(:,:,:,2);
    test_accuracy = mean(test_accuracy, 3);
    figure;
    subplot(1,2,1);
    surf(train_accuracy);
    title('train');
    subplot(1,2,2);
    surf(test_accuracy);
    title('test');
end
