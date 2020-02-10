function W = flda(data, labels)

% compute dimensionality of samples
d = size(data, 2);

% find number of classes
classes = unique(labels);
k = numel(classes);

% check that the dimensionality is greater than the number of classes
%assert(d >= k);

% make sure we have more samples than dimensions for each class
%for i = 1:k
%    assert(d <= size(find(labels == classes(i)), 1), 'One class does not have enough samples');
%end

% compute S_w
S_w = zeros(d, d);
for i = 1:k
    inds = find(labels == classes(i));
    m_i = mean(data(inds, :));

    S_i = zeros(d, d);
    for j = 1:numel(inds)
        S_i = S_i + (data(inds(j), :) - m_i)' * (data(inds(j), :) - m_i);
    end
    S_w = S_w + S_i;
end

% compute S_b
S_b = zeros(d, d);
m = mean(data);
for i = 1:k
    inds = find(labels == classes(i));
    m_i = mean(data(inds, :));
    N_k = numel(inds);
    S_b = S_b + N_k .* (m_i - m)' * (m_i - m);
end

% compute eigenvectors of S_w^-1 * S_B
[W, D] = eigs(S_w \ S_b, k-1);

end
