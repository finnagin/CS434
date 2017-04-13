function [tr_error, te_error, w] = sse(Xtr, Ytr, Xte, Yte, lambda)
% NOTE: 
% Using '\' and the build in command
% inv() produce completely different
% usm of sqyare errors. 
% When using '\', we obtain 
% errors with 10^10; when using inv(), 
% errors are less than 10^4.

% compute optimal weight
Ttr = Xtr'*Xtr; 
m = size(Ttr, 1);
I = eye(m);
w = pinv(Ttr + lambda*I)*(Xtr'*Ytr);

% compute sum of square error
h = Xtr*w; % hypothesis function (model)

% sum of square error of training data
tr_error = (Ytr - h)'*(Ytr - h); 
% sum of square error of testing data
te_error = (Yte - Xte*w)'*(Yte - Xte*w); 
end 