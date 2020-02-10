function [rate, time] = classify_flda(data, labels, testData, testLabels, metric)

  if strcmp(metric, 'manhattan')
    distFunc = @manhattan;
  elseif strcmp(metric, 'euclidean')
    distFunc = @euclidean;
  elseif strcmp(metric, 'mahalanobis')
    distFunc = @mahalanobis;
  end

  % compute alpha and means
  disp('computing flda projection');
  t = cputime;
  W = flda(data, labels);
  time = cputime - t;

  % compute projected means
  disp('computing projected means');
  k = numel(unique(labels));
  means = zeros(k-1, k);
  for i = 1:k
      means(:, i) = mean(W' * data(find(labels == i), :)', 2);
  end
  
  % compute projected covariances
  disp('computing projected covariances');
  covars = zeros(k-1, k-1, k);
  for i = 1:k
      covars(:,:,i) = cov((W' * data(find(labels == i), :)')');
  end
  
  % compute priors
  disp('computing priors');
  priors = zeros(k);
  N = length(labels);
  for i = 1:k
      priors(i) = numel(find(labels == i)) / N;
  end
  
  % classify each samples
  disp('classifying...');
  num_correct = 0;
  num_tests = numel(testLabels);
  for i = 1:num_tests;
    y = W' * testData(i,:)';
    
    ind = distFunc(y, means, covars, priors);
    if ind == testLabels(i)
        num_correct = num_correct + 1;
    end
  end

  rate = num_correct / num_tests;
end

function ind = euclidean(y, means, covars, priors)
    k = size(means, 2);
    d = zeros(1, k);
    for i = 1:k
      d(i) = norm(y - means(:,i), 2);
    end
    [m, ind] = min(d);
end

function ind = manhattan(y, means, covars, priors)
    k = size(means, 2);
    d = zeros(1, k);
    for i = 1:k
      d(i) = norm(y - means(:,i), 1);
    end
    [m, ind] = min(d);
end

function ind = mahalanobis(y, means, covars, priors)
    k = size(means, 2);
    d = zeros(1, k);
    for i = 1:k
        sigma = covars(:, :, i);
        mu = means(:, i);
        d(i) = -0.5 * log(det(sigma)) - 0.5 * (y - mu)' * inv(sigma) * (y - mu) + log(priors(i));
    end
    [m, ind] = max(d);
end
