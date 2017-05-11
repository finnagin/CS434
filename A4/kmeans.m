function [groups, centers] = kmeans(X, k)
%% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % k - A vector containing the class to be predicted for each sample
    
%% Output:
    % SSE - The feature to split the data on
    % groups

                
%% code:
n = size(X,1);


centers = X(randperm(n,k),:);

for a = 1:10
    groups = cell(k,1);
    for b = 1:n
        dist = [inf 0];
        for c = 1:k
            if norm(centers(c,:)-X(b,:))^2 < dist(1)
                dist = [norm(centers(c,:)-X(b,:))^2 c];
            end
        end
        groups{dist(2)}(end+1,:) = X(b,:);
    end
    for c = 1:k
        centers(c,:) = sum(X)/size(groups{c},1);
    end
end





end