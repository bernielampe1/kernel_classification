function [delta, labels] = genDelta(thetas, d, factor)

%factor = 200;

% number of observations to generate
N_delta = (1+2+3+4+5) * factor;

% allocate space for delta
delta = zeros(N_delta, d);

% generate the class labels
labels = zeros(N_delta, 1);

% generate k*200 samples for each theta each of dimension d
i = 1;
for k = 1:length(thetas)
    N = k * factor;
    for n = 1:N
        delta(i,:) = genObservation(d, thetas(k));
        labels(i) = k;
        i = i + 1;
    end
end

end
