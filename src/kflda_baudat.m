function [alpha, K] = kflda_baudat(data, labels, kernel, kparams, lambda)

% based on the paper by Baudat and Anouar named
% "Generalized Discriminant Analysis Using a Kernel Approach"

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

% Sort data according to labels
[foo, ind] = sort(labels);
labels = labels(ind);
data = data(ind,:);

% Compute kernel matrix
disp('computing kernel matrix...');
K = gram(data, data, kernel, kparams);

% Construct diagonal block matrix W
W = [];
for i=1:c
    num_data_class = length(find(labels == i));
    W = blkdiag(W, ones(num_data_class) / num_data_class);
end

% Compute centering matrix
%N = size(data, 1);
%D = sum(K) / N;
%E = sum(D) / N;
%J = ones(N, 1) * D;
%K = K - J - J' + E * ones(N, N);

% Perform eigenvector decomposition of kernel matrix (Kc = P * gamma * P')
disp('performing eigendecomposition of kernel matrix...');
[P, gamma] = eig(K);

% Sort eigenvalues and vectors in descending order
[gamma, ind] = sort(diag(gamma), 'descend');
%P = P(:,ind);

% Remove eigenvectors with relatively small value
%minEigv = max(gamma) / 1e5;
%ind = find(gamma > minEigv);
%P = P(:,ind);
%gamma = gamma(ind);

% Perform eigendecomposition of matrix (P' * W * P)
[Beta, D] = eigs(P' * W * P, c-1);

% Compute final embedding alpha
alpha = P * diag(1 ./ gamma) * Beta;

% Normalize embedding
for i=1:c-1
    alpha(:,i) = alpha(:,i) / sqrt(alpha(:,i)' * K * alpha(:,i));
end
