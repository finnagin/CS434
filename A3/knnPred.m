function P = knnPred(Xr, Yr, Xe, K)
%% Parameters:
    % Xr - An array containing the training feature vectors from each of the samples
    % Yr - A vector containing the training class to be predicted for each sample
    % Xe - An array containing the feature vectors from each of the samples
    %      for prediction
    % K - the number of neighbors to consider

%% code:

P = zeros(size(Xe, 1),1); % initialize the vector of predictions

for a = 1:size(Xe, 1) % iterate over the testing data
    nord = [(1:size(Xr, 1))' zeros(size(Xr, 1),1)]; % initialize a vector for norms and indicies
    for b = 1:size(Xr, 1) % iterate over the training data
        nord(b,2) = norm(Xr(b,:) - Xe(a,:)); % calculate the distance from the point to predict on for each training data point
    end
    nord = sortrows(nord,2); % sort by distance
    ord = nord(1:K,1); % extract the closest points
    P(a) = mode(Yr(ord)); % make prediction by choosing the point that shows up most
    
end