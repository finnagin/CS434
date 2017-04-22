function L = LossFunc(X, Y, w)
% loss function

h = sigmoid(X*w);
L = sum(-log(h(Y == 1))) + sum(-log(1 - h(Y == 0)));

end
