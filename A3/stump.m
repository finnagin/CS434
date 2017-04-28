function branch = stump(X, Y)
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

info = [zeros(1,size(X,2));1:size(X,2)]';

for n = 1:size(X,2)
    gain = 1.5;
    X_sorted = sortrows(X,n);
    for c = 2:size(X,1)
        Y_old = Y(c-1);
        Y_new = Y(c);
        if Y_old ~= Y_new
            p1 = sum(Y(1:c-1) == 1)/(c-1);
            p2 = sum(Y(c:length(Y)) == 1)/(leangth(Y)-c+1);
            gain_c = -(c-1)/length(Y)*(p1*log2(p1)+(1-p1)*log2(1-p1)) +...
                        -(1-(c-1)/length(Y))*(p2*log2(p2)+(1-p2)*log2(1-p2))
            if gain_c < gain
                gain = gain_c;
                split = [n (X_sorted(c,n)+X_sorted(c-1,n))/2];
            end
        end
    end
    info(n,:) = split
end

info = sortrows(info,2)
branch = info(1,:)
end
