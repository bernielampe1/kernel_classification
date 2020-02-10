function [rate, time, class_rates] = classify_ksvm(data, labels, test_data, test_labels, kernelName, kernelParams)

    t = cputime();
    
    % compute Gram matrix
    K = gram(data, data, kernelName, kernelParams);

    % compute svm model
    n = size(data, 1);
    K1 = [(1:n)', K];
    model = svmtrain(labels, K1, '-q -t 4');
    time = cputime() - t;
    
    % compute Gram test data matrix
    K = gram(test_data, data, kernelName, kernelParams);
    
    % run classifications
    n = size(test_data, 1);
    K1 = [(1:n)', K];
    y = svmpredict(test_labels, K1, model, '-q');
    
    % compute classification rate
    num_correct = numel(find((y - test_labels) == 0));
    rate = num_correct / numel(test_labels);
    
    k = numel(unique(test_labels));
    class_rates = zeros(k, 1);
    for i = 1:k
        inds = find(test_labels == i);
        n = numel(find((y(inds) - test_labels(inds)) == 0));
        class_rates(i) = n / numel(inds);
    end
end
