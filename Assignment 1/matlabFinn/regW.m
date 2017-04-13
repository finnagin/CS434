function a = regW(X, Y, l)

a = pinv(X'*X+l*eye(size(X'*X,1)))*(X'*Y);


end