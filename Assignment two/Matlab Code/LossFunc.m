function L = LossFunc(X, Y, w)
% loss function

h = sigmoid(X*w);
L = -Y'*log(h) - (1 - Y')*log(1 - h);

end