function [feature, theta, info_gain, l_X, l_Y, r_X, r_Y, l_flag, r_flag] = stump(X, Y)
%% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % Y - A vector containing the class to be predicted for each sample
    % N - The number of iterations
    % learn - the learning rate
    % Lam - the lambda used for regularization
    %       (if no regularization is desired use Lam = 0)
    % loss - A optional indicator for the type of output desired
                % If loss != 1 or is not inputed then the end W vector is
                % outputted
                % If loss = 1 then instead of the W being outputted a
                % vector of the loss function for every iteration is
                % outputted
                
%% code:
%      [1 features |  2 split value   |   3 info_gain    ]
info = [1:size(X,2);zeros(1,size(X,2));zeros(1,size(X,2))]';

for n = 1:size(X,2)
    gain = -100;
    X_sorted = sortrows([X Y],n);
    Y_sorted = X_sorted(:,end);
    X_sorted = X_sorted(:,1:(end-1));
    split = [0 -100];
    
    for c = 2:size(X,1)
        Y_old = Y_sorted(c-1);
        Y_new = Y_sorted(c);
        if Y_old ~= Y_new
            p1 = sum(Y(1:c-1) == 1)/(c-1);
            p2 = sum(Y(c:length(Y)) == 1)/(length(Y)-c+1);
            p = sum(Y == 1)/length(Y);
            gain_c = -p*mylog2(p) - (1-p)*mylog2(1-p) -...
                        (-(c-1)/length(Y)*(p1*mylog2(p1)+(1-p1)*mylog2(1-p1)) +...
                        -(1-(c-1)/length(Y))*(p2*mylog2(p2)+(1-p2)*mylog2(1-p2)));
            if gain_c > gain
                gain = gain_c;
                split = [(X_sorted(c,n)+X_sorted(c-1,n))/2 gain_c];
            end
        end
    end
    info(n,2:3) = split;
end

info = sortrows(info,3);
theta = info(1,2);
feature = info(1,1);
split_ind = X(:,feature) < theta;
l_X = X(split_ind,:);
r_X = X(~split_ind,:);
l_Y = Y(split_ind);
r_Y = Y(~split_ind);
if sum(l_Y == 1) == length(l_Y) || sum(l_Y == 1) == 0
    l_flag = true;
else
    l_flag = false;
end

if sum(r_Y == 1) == length(r_Y) || sum(r_Y == 1) == 0
    r_flag = true;
else
    r_flag = false;
end

info_gain = gain;

end
