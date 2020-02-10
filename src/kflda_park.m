function [alpha, K] = kflda_park(data, labels, kernelName, kernelParams, lambda) % lambda is not used

% based on the paper by Cheong Hee Parka nd Haesun Park
% "Nonlinear Discriminant Analysis Using Kernel Functions and the 
% Gneralized Singular Value Decomposition"

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
disp('computing gram matrix');
K = gram(data, data, kernelName, kernelParams);

% compute K_w and K_b matricies
disp('computing K_w and K_b');
n = size(data, 1);
K_w = zeros(n, n);
K_b = zeros(n, c);
for i = 1:c
    inds = find(labels == i);
    l = numel(inds);
    K_w(:, inds) = 1/l * sum(K(:, inds),2) * ones(1, l);
    K_b(:, i) = sqrt(l) * (1/l * sum(K(:, inds), 2) - 1/n * sum(K, 2));
end
K_w = K * K - K_w;

% apply gsvd to (K_b', K_w') pair
disp('computing gsvd');
[U,V,X,C,S] = gsvd(K_b', K_w');

% alpha is the first k-1 columns
alpha = X(:, 1:c-1);

end
