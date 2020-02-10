function [rate, time, class_rates] = classify_kflda(data, labels, testData, testLabels, kflda_handle, metric, kernel, kparams)

  if strcmp(metric, 'manhattan')
    distFunc = @manhattan;
  elseif strcmp(metric, 'euclidean')
    distFunc = @euclidean;
  elseif strcmp(metric, 'mahalanobis')
    distFunc = @mahalanobis;
  end

  % regularization parameter
  lambda = 1e15;
  
  % compute alpha and means
  disp('computing kflda projection');
  t = cputime;
  [alpha, K] = kflda_handle(data, labels, kernel, kparams, lambda);
  time = cputime - t;   
  
  % compute projected means
  disp('computing projected means');
  k = numel(unique(labels));
  means = zeros(k-1, k);
  for i = 1:k
      inds = find(labels == i);
      means(:, i) = mean(alpha' * K(:, inds), 2);
  end
  
  % compute projected covariances
  disp('computing projected covariances');
  covars = zeros(k-1, k-1, k);
  for i = 1:k
      inds = find(labels == i);
      covars(:, :, i) = cov((alpha' * K(:, inds))');
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
  class_rates = zeros(k, 1);
  num_tests = numel(testLabels);
 % ys = zeros(num_tests,3);
  for i = 1:num_tests;
    y = alpha' * gram(data, testData(i,:), kernel, kparams);
 %   ys(i,:) = y;
    ind = distFunc(y, means, covars, priors);
    if ind == testLabels(i)
        num_correct = num_correct + 1;
        class_rates(testLabels(i)) = class_rates(testLabels(i)) + 1;
    end
  end
  
%  figure;
%  plot3(ys);

  rate = num_correct / num_tests;
  for i = 1:k
     class_rates(i) = class_rates(i) / numel(find(testLabels == i)); 
  end
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
