function tree = treePlanter(X, Y, layers)
%% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % Y - A vector containing the class to be predicted for each sample
    % layers - the maximum depth desired
                
                
%% Output:            
% tree - | 1 feature | 2 theta | 3 info gain | 4 left flag | 5 right flag |
% 6 left child | 7 right child | 8 parent | 9 left prediction | 10 right prediction |
%% code:
if ~(isequal(fix(layers),layers) && layers > .5) % check to make sure layers were input correctly
    layers = 1;
    warning('Layers imput must be a positive integer. Setting to 1...')
end

tree = zeros(sum(2.^(0:(layers-1))), 10); % initialize the tree
skip = 0; % set skip counter to zero
data = cell(2,sum(2.^(0:(layers-1)))); % initialize data cell array

data{1,1} = X; % input X into data cell
data{2,1} = Y; % input Y into data cell

c = 1; % initialize the child counter
for k = 1:sum(2.^(0:(layers-1))) % interate over all possible nodes
    if (k + skip) > sum(2.^(0:(layers-1))) || c < k % check if all nodes are finished
        for g = 1:length(tree(:,6)) % trim tree
            if tree(g,6) > k-1
                tree(g,6) = 0;
            end
        end
        for g = 1:length(tree(:,7)) % remove non existent child branches
            if tree(g,7) > k-1
                tree(g,7) = 0;
            end
        end
        tree = tree(1:(k-1),:); % return tree
        return
    end
    
%    if side(k) == 1
%        test = Xi(:,tree(tree(k,8),1)) <= tree(tree(k,8),2);
%        Xi = Xi(test,:);
%        Yi = Yi(test);
%    elseif side(k) == 2
%        test = Xi(:,tree(tree(k,8),1)) > tree(tree(k,8),2);
%        Xi = Xi(test,:);
%        Yi = Yi(test);
%    end
    
    [tree(k,1), tree(k,2), tree(k,3), l_X, l_Y, r_X, r_Y, tree(k,4), tree(k,5)] = stump(data{1,k}, data{2,k});
    
    if tree(k,3) == -100 % this was to indicate I was getting a certain error
        tree(k, 9) = tree(tree(k,8),9);
        tree(k, 10) = tree(tree(k,8),10);
        tree(k,4) = true;
        tree(k,5) = true;
        warning('no split found...')
    end
    
    
    if sum(l_Y == 1) > sum(l_Y == -1) % make prediction at left node
        tree(k,9) = 1;
    elseif sum(l_Y == 1) < sum(l_Y == -1)
        tree(k,9) = -1;
    else
        warning('even split')
        tree(k,9) = randsample([-1 1], 1);
    end
    
    if sum(r_Y == 1) > sum(r_Y == -1) % make prediction at left node
        tree(k,10) = 1;
    elseif sum(r_Y == 1) < sum(r_Y == -1)
        tree(k,10) = -1;
    else
        warning('even split')
        tree(k,10) = randsample([-1 1], 1);
    end
    
    if ~tree(k,4) % check leaf flag
        c = c+1; % count child
        if c <= sum(2.^(0:(layers-1))) - skip % make sure we don't have too many nodes
            data{1,c} = l_X; % store child data
            data{2,c} = l_Y;
            tree(k,6) = c; % srore child pointer
            tree(c,8) = k; % srore child's parent pointer
        end
    else
        skip = skip + 1; % add to skip counter
    end
    
    if ~tree(k,5) % check leaf flag
        c = c+1; % count child
        if c <= sum(2.^(0:(layers-1))) - skip % make sure we don't have too many nodes
            data{1,c} = r_X; % store child data
            data{2,c} = r_Y;
            tree(k,7) = c; % srore child pointer
            tree(c,8) = k; % srore child's parent pointer
        end
    else
        skip = skip + 1; % add to skip counter
    end
    
end







end