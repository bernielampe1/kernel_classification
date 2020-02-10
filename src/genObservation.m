function X = genObservation(N, p)
    X = zeros(1,N);
    for i = 1:N
        if rand < p
            X(i) = 1;
        else
            X(i) = 0;
        end
    end
end