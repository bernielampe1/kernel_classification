function time = time_kernels(data, kernel, kparams)

t = cputime();
% Compute kernel matrix
disp('computing kernel matrix...');
gram(data, data, kernel, kparams);
time = cputime() - t;

end