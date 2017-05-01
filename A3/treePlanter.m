function tree = treePlanter(X, Y, layers)
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
                
                
                
% tree - | 1 feature | 2 theta | 3 info gain | 4 left flag | 5 right flag |
% 6 left child | 7 right child | 8 parent | 9 left prediction | 10 right prediction |
%% code:
if ~(isequal(fix(layers),layers) && layers > .5)
    layers = 1;
    warning('Layers imput must be a positive integer. Setting to 1...')
end

tree = zeros(sum(2.^(0:(layers-1))), 12);
skip = 0;
data = cell(2,sum(2.^(0:(layers-1))));

data{1,1} = X;
data{2,1} = Y;

c = 1;
for k = 1:sum(2.^(0:(layers-1)))
    if (k + skip) > sum(2.^(0:(layers-1)))
        tree = tree(1:(k-1),:);
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
    
    tree(k,11) = length(l_Y);
    tree(k,12) = length(r_Y);
    
    if tree(k,3) == -100
        tree(k, 9) = tree(tree(k,8),9);
        tree(k, 10) = tree(tree(k,8),10);
        tree(k,4) = true;
        tree(k,5) = true;
        warning('no split found...')
    end
    
    
    if sum(l_Y == 1) > sum(l_Y == -1)
        tree(k,9) = 1;
    elseif sum(l_Y == 1) < sum(l_Y == -1)
        tree(k,9) = -1;
    else
        warning('even split')
        tree(k,9) = randsample([-1 1], 1);
    end
    
    if sum(r_Y == 1) > sum(r_Y == -1)
        tree(k,10) = 1;
    elseif sum(r_Y == 1) < sum(r_Y == -1)
        tree(k,10) = -1;
    else
        warning('even split')
        tree(k,10) = randsample([-1 1], 1);
    end
    
    if ~tree(k,4)
        c = c+1;
        if c <= sum(2.^(0:(layers-1))) - skip
            data{1,c} = l_X;
            data{2,c} = l_Y;
            tree(k,6) = c;
            tree(c,8) = k;
        end
    else
        skip = skip + 1;
    end
    
    if ~tree(k,5)
        c = c+1;
        if c <= sum(2.^(0:(layers-1))) - skip
            data{1,c} = r_X;
            data{2,c} = r_Y;
            tree(k,6) = c;
            tree(c,8) = k;
        end
    else
        skip = skip + 1;
    end
    
end







end