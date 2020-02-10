function G = gram(X1, X2, kernel, params)

assert(size(X1, 2) == size(X2, 2), 'dimensionality of both datasets should be equal');
    
switch kernel      
  % Linear kernel
  case 'linear'
    param1 = params(1);
    G = X1 * X2' + param1;
    
  % Sigmoid kernel
  case 'sigmoid'
    param1 = params(1);
    param2 = params(2);
    G = tanh(param1 * (X1 * X2') + param2);
      
  % RBF kernel
  case 'rbf'
    param1 = params(1);
    G = EuDist2(X1, X2);
    G = exp(-(G.^2 / (2 * param1.^2)));
    
  % Polynomial kernel
  case 'poly'
    param1 = params(1);
    param2 = params(2);
    param3 = params(3);
    G = (param1 * (X1 * X2') + param2) .^ param3;  
  
  otherwise
    assert(0 == 1, 'unknown kernel parameter name');
end

end
