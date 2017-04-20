function g = sigmoid(x)
% Compute the sigmoid function 

[m, n] = size(x); % obtain the size of x
g = zeros(m, n); % initialize g

g = 1./(1 + exp(-x));

end