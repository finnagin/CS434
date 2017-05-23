function [group, sse] = Kmeans(X, K)
%% Parameters:
    % X - An array containing the data
    % K - The number of clusters desired
        
%% Outputs:
    % group - the output of clusters
    %         (note: Thisi s a vector of integers where the integer
    %         corresponds to the cluster the example of the same index in
    %         the array X)
    % sse - The sum of squared errors

%% code:
samples = size(X, 1); % finds the size of the data
iter = 0; % initialize iteration counter
center = X(randperm(samples, K), :); % This initialized the centers from random 
                                     % points in the data
idx = zeros(samples, 1); %initializes the index vector
sse = [0 1]; % initializes the sse (sse(1) = last interations see & 
             %                      sse(2) = this iterations sse)

while abs(sse(end) - sse(end-1)) > 1e-8 && iter < 5000 % check for convergence
    error = 0; % initialize error counter
    for s = 1:samples % iterate through all samples
        distance = zeros(K, 1); % initialize distance vector
        for c = 1:K % iterate through clusters
            distance(c, 1) = norm(X(s,:) - center(c,:)); % calculate distance for each cluster
        end
        [minDist, minIdx] = min(distance); % find minimum distance
        idx(s) = minIdx; % set index to index coresponding to the minimum distance
    end
    for k = 1:K % iterate through clusters
        pos = find(idx == k); % find the examples belonging to this cluster
        if size(pos) ~= 0 % check if cluster is empty (error prevention)
            center(k,:) = mean(X(pos,:)); % find the new center
            error = error + sum(sum((X(pos,:) - repmat(center(k,:),length(pos),1)).^2)); % find the sse of new clusters
        end
    end
    sse = [sse, error]; % set new sse and move old one over
    iter = iter + 1; % count iterations
end

sse = sse(3:end); % return sse
group = idx; % return group
end