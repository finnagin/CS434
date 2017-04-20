function w = Batgrad(X, Y, N, w0, learning_rate, lambda)
% Apply Batch gradient decent
%
% In N interation, if the loss is less than the setting threshold
% stop computing the optimal weight.
%
% initialize optimal weight vector
%

[samples, features] = size(X);
w = w0; %obtain initial w
wNorm = zeros(N, 1); % store the norm of weight in each interation

for iter = 1:N
    delta = zeros(features, 1); % initialize error
    
    for n = 1:samples
        h = sigmoid(X(n, :)*w); % hypothese function  
        delta = delta + (h - Y(n))*X(n, :)';
    end
    w = w - learning_rate*(delta + lambda*sqrt(w'*w)); % update optimal weight vector
    wNorm(iter) = norm(w, 2); 
end

% figure
% plot(1:N, wNorm, '-')
% ylabel('Weight Norm')
% xlabel('Iteration')
% title('Batch Gradient Decent')
% hold off

end