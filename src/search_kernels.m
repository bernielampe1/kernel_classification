function [rates_poly, rates_sig, rates_rbf] = search_kernels(data, labels, test_data, test_labels)

    algs = {@kflda_mika, @kflda_max, @kflda_baudat, @kflda_park};

    % KFLDA poly kernel
    rates_poly = zeros(numel(algs)+1, 10);
    for a = 1:numel(algs)
       for i = 1:10
           rates_poly(a, i) = classify_kflda(data, labels, test_data, test_labels, algs{a}, 'mahalanobis', 'poly', [1 0 i])
       end
    end

    % KFDLA sig kernel
    rates_sig = zeros(numel(algs)+1, 10);
    for a = 1:numel(algs)
        k = 1;
        for i = 0.01:0.05:1.1
            rates_sig(a, k) = classify_kflda(data, labels, test_data, test_labels, algs{a}, 'mahalanobis', 'sigmoid', [i 0])
            k = k + 1;
        end
    end

    % KFDLA rbf kernel
    rates_rbf = zeros(numel(algs)+1, 10);
    for a = 1:numel(algs)
        for i = 1:10
            rates_rbf(a, i) = classify_kflda(data, labels, test_data, test_labels, algs{a}, 'mahalanobis', 'rbf', i)
        end
    end
    
    % SVM poly kernel
    for i = 1:10
        rates_poly(5, i) = classify_ksvm(data, labels, test_data, test_labels, 'poly', [1 0 i])
    end
    % SVM sig kernel
    k = 1;
    for i = 0.01:0.05:1.1
        rates_sig(5, k) = classify_ksvm(data, labels, test_data, test_labels, 'sigmoid', [i 0])
        k = k + 1;
    end
    % SVM RBF kernel
    for i = 1:10
        rates_rbf(5, i) = classify_ksvm(data, labels, test_data, test_labels, 'rbf', i)
    end
end
