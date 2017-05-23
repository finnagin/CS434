function [group, clust, infoMatrix] = SingleL(X, K)
%% Parameters:
    % X - An array containing the data
    % K - The number of desired vectors
        
%% Outputs:
    % clust - the cluster groupings after K clusters are left
    % infoMatrix - the distance matrix
    % group - a vector of cluster names (integers) that correspond to
    %         each example X of the same index #

%% code:
samples = size(X, 1);
clust = []; % initializing cluster
c = samples; % initializing new number cluster
distance = pdist(X); % compute pair wise distance
DistValue = squareform(distance); % comstruct a matrix for each data point
DistMatrix = DistValue;
groupTemp = 1:samples; % initialize group

% comstruct a matrix to store all information
infoMatrix = zeros(samples + 1, samples + 1); 
infoMatrix(1, :) = 0:samples;
infoMatrix(2:end, 1) = 1:samples;
infoMatrix(2:end, 2:end) = DistMatrix;

while length(DistMatrix) > 2
    [minD, pos]= min(distance); % find the minimum distance
    DistN = DistMatrix; % rename distance matrix
    DistN(triu(true(size(DistN)))) = NaN; % avoid duplicate distance
    % find the index of the min distance
    [min_i, min_j] = find(DistN == minD,1); 
%     fprintf(['Min distance between ',num2str(min_i),' and ',...
%         num2str(min_j), ' is ', num2str(minD),'\n'])
    
    if size(min_i) ~= 0 % to make sure obtain index of min distance 
        c = c + 1; % update cluster
        % append new cluster
        if length(DistMatrix) < K+1
            clust = [clust; {infoMatrix(1, min_i + 1), ...
                infoMatrix(1, min_j + 1), minD}]; 
        end
        % update cluster number to infoM
        groupTemp(groupTemp == infoMatrix(1, min_j + 1)) = c;
        groupTemp(groupTemp == infoMatrix(1, min_i + 1)) = c;
        infoMatrix(1, min_j + 1) = c;
        infoMatrix(min_j + 1, 1) = c;
        % Complete link variation:
        % max_dist = max(DistM(:,min_i), DistM(:,min_j));
        % Single Link Variation:
        min_dist = min(DistMatrix(:,min_i), DistMatrix(:,min_j));
        min_dist(min_j) = 0;
        % update new distance to cluster
        DistMatrix(:,min_j) = min_dist;
        DistMatrix(min_j,:) = min_dist';
    end
    
    % delete old distance
    DistMatrix(min_i,:) = [];
    DistMatrix(:,min_i) = [];
    infoMatrix(min_i + 1,:) = [];
    infoMatrix(:,min_i + 1) = [];
    distance(pos) = [];
    infoMatrix(2:end, 2:end) = DistMatrix;
    
    % check # of clusters
    if length(DistMatrix) == K
        group = groupTemp; % return grouping with K clusters
    end
end

end