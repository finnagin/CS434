function L = lossFun(X, Y, W)
    h = 1./(1+exp(-W*X'));
    Y_bool = Y == 1;
    %L = -Y'*log(h)' - (1 - Y')*log(1 - h)';
    L = sum(-log(h(Y_bool))) + sum(-log(1-h(~Y_bool)));
end