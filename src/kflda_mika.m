function [alpha, K] = kflda_mika(data, labels, kernelName, kernelParams, lambda)

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

% loop over classes and compute N and M
n = size(data, 1);
N = zeros(n, n);
M = zeros(n, n);
M_s = 1/n * sum(K, 2);
for i = 1:c
   inds = find(labels == i);
   l = numel(inds);
   M_i = 1/l .* sum(K(:, inds), 2);
   M = M + l .* ((M_i - M_s) * (M_i - M_s)');

   K_i = K(:, inds);
   N = N + K_i * (eye(l) - 1/l * ones(l)) * K_i';
end
N = N + lambda * eye(n);

% get c-1 leading eigenvectors
[alpha, D] = eigs(N \ M, c-1);

end
