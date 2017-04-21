function L = lossFun(X, Y, W)
%% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % Y - A vector containing the class to be predicted for each sample
    % W - weight vector for predictions
                
%% code:


    h = 1./(1+exp(-W*X')); % makes the prediction
    Y_bool = Y == 1; % changes class to a logical
    % (This is to prevent a NaN result for a machine accuracy error)
    
    %L = -Y'*log(h)' - (1 - Y')*log(1 - h)';
    L = sum(-log(h(Y_bool))) + sum(-log(1-h(~Y_bool)));
    % Notice that instead of -Y*log(...) I indexed the h vector by the
    % indicies that Y is 0 or 1 resulting in the same sum since all the
    % terms will go to zero where Y != 1 (or zero for the 2nd sum)
end