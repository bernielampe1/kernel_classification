function gram_test(data)
    param = 0.1;

    % test rbf
    K = gram(data, data, 'rbf', param);
    
    n = size(data, 1);
    Kp = zeros(n);
    for i = 1:n
        for j = 1:n
            x = data(i, :);
            y = data(j, :);
            Kp(i,j) = norm(x - y, 2);
        end
    end
    Kp = exp(-(Kp.^2 / (2 * param.^2)));
    
    isequal(K, Kp)
    
    % test linear
    param = 1;

    K = gram(data, data, 'linear', param);
    
    n = size(data, 1);
    Kp = zeros(n);
    for i = 1:n
        for j = 1:n
            x = data(i, :);
            y = data(j, :);
            Kp(i,j) = x * y' + param;
        end
    end
    
    isequal(K, Kp)
    
    % test poly
    param = [1 1 4];

    K = gram(data, data, 'poly', param);
    
    n = size(data, 1);
    Kp = zeros(n);
    for i = 1:n
        for j = 1:n
            x = data(i, :);
            y = data(j, :);
            Kp(i,j) = (param(1) * x * y' + param(2)).^param(3);
        end
    end

    isequal(K, Kp)
    
    % test sigmoid
    param = [2 2];

    K = gram(data, data, 'sigmoid', param);
    
    n = size(data, 1);
    Kp = zeros(n);
    for i = 1:n
        for j = 1:n
            x = data(i, :);
            y = data(j, :);
            Kp(i,j) = tanh(param(1) * (x * y') + param(2));
        end
    end
    
    isequal(K, Kp)
end
