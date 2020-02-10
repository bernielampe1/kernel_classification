function [rate, time] = classify_svm(data, labels, test_data, test_labels)

    % compute svm model
    t = cputime();
    modelparams = '-c 1 -q';
    model = svmtrain(labels, data, modelparams);
    time = cputime() - t;
    
    % run classifications
    y = svmpredict(test_labels, test_data, model, '-q');
    
    % compute classification rate
    num_correct = numel(find((y - test_labels) == 0));
    rate = num_correct / numel(test_labels);
end
