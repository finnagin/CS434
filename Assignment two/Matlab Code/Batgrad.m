function output = Batgrad(X, Y, N, w0, learning_rate, lambda, options)
% Apply Batch gradient decent
%
% In N interation, if the loss is less than the setting threshold
% stop computing the optimal weight.
% options = 1, output is loss
% options = 0, output is w
%
% initialize optimal weight vector
%

[samples, features] = size(X);
w = w0; %obtain initial w
loss = zeros(N, 1); 
for iter = 1:N
    delta = zeros(features, 1); % initialize error
    
    for n = 1:samples
        h = sigmoid(X(n, :)*w); % hypothese function  
        delta = delta + (Y(n) - h)*X(n, :)';
    end
    w = w + learning_rate*(delta + lambda*w); % update optimal weight vector
    loss(iter) = LossFunc(X, Y, w);
end

if options == 1
    output = loss;
end
if options == 0
    output = w;
end
end
                
