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

info = [1:size(X,2);zeros(1,size(X,2));zeros(1,size(X,2))]';

for n = 1:size(X,2)
    gain = 1.5;
    X_sorted = sortrows([X Y],n);
    Y_sorted = X_sorted(:,end);
    
    for c = 2:size(X,1)
        Y_old = Y_sorted(c-1);
        Y_new = Y_sorted(c);
        if Y_old ~= Y_new
            p1 = sum(Y(1:c-1) == 1)/(c-1);
            p2 = sum(Y(c:length(Y)) == 1)/(length(Y)-c+1);
            gain_c = -(c-1)/length(Y)*(p1*log2(p1)+(1-p1)*log2(1-p1)) +...
                        -(1-(c-1)/length(Y))*(p2*log2(p2)+(1-p2)*log2(1-p2));
            if gain_c < gain
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
if sum(l_Y == 1) == length(l_Y)
    l_flag = true;
else
    l_flag = false;
end

if sum(r_Y == 1) == length(r_Y)
    r_flag = true;
else
    r_flag = false;
end

info_gain = info;

end
