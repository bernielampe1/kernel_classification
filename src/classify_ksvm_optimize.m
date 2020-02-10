    function [rates, times, class_rates] = classify_ksvm_optimize(data, labels, test_data, test_labels, kernel, kparams)
    k = 1;
    rates = zeros(1, 10);
    times = zeros(1, 10);
    nclasses = numel(unique(labels));
    class_rates = zeros(nclasses, 10);
    for p = 0.05:0.05:0.5
        % build data and labels
        p_data = [];
        p_labels = [];
        for c = 1:nclasses
            inds = find(labels == c);
            p_inds = randsample(inds, ceil(p*numel(inds)));
            p_data = [p_data; data(p_inds, :)];
            p_labels = [p_labels; labels(p_inds)];
        end

        [rates(k), times(k), class_rates(:,k)] = classify_ksvm(p_data, p_labels, test_data, test_labels, kernel, kparams)
        k = k + 1;
    end
end
