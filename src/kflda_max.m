function [alpha, K] = kflda_max(data, labels, kernelName, kernelParams, lambda)

% based on the paper by Max Welling, "Fisher Linear Discriminant Analysis"

% compute dimensionality of samples
d = size(data, 2);

% find number of classes
classes = unique(labels);
c = numel(classes);

% check that the dimensionality is greater than the number of classes
%assert(d >= c, 'dimensionality is less than the number of classes');

% make sure we have more samples than dimensions for each class
%for i = 1:c
%    assert(d <= size(find(labels == classes(i)), 1), 'one class does not have enough samples');
%end

% compute Gram matrix
K = gram(data, data, kernelName, kernelParams);

% compute k_c and k
n = size(data, 1);
k = 1/n * sum(K, 2);
S_b = zeros(n, n);
S_w = zeros(n, n);
for i = 1:c
    inds = find(labels == i);
    n_c = numel(inds);
    k_c = 1/n_c * sum(K(:, inds), 2);
    
    S_b = S_b + k_c * k_c' - k * k';
    S_w = S_w + n_c * (k_c * k_c');
end
S_w = K * K - S_w + lambda * eye(n);

% get eigenvectors
[alpha, D] = eigs(S_w \ S_b, c-1);

end
