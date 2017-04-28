function P = knnPred(Xr, Yr, Xe, K)
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

P = zeros(size(Xe, 1),1);

for a = 1:size(Xe, 1)
    nord = [(1:size(Xr, 1))' zeros(size(Xr, 1),1)];
    for b = 1:size(Xr, 1)
        nord(b,2) = norm(Xr(b,:) - Xe(a,:));
    end
    nord = sortrows(nord,2);
    ord = nord(1:K,1);
    P(a) = mode(Yr(ord));
    
end