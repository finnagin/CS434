function a = solveW(X, Y)

a = pinv(X'*X)*(X'*Y);


end