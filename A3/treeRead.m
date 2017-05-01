function pred = treeRead(X_Test, tree)
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

%% Code:

pred = zeros(1, size(X_Test,1))';

for k = 1:size(X_Test,1)
    c = 0;
    j = 1;
    while c <= size(tree,1)
        c = c+1;
        if X_Test(k,tree(j,1)) < tree(j,2)
            j_new = tree(j, 6);
            if j_new == 0
                pred(k) = tree(j, 9);
                c = size(tree,1) + 1;
            else
                j = j_new;
            end
            
        end
        
        if X_Test(k,tree(j,1)) > tree(j,2)
            j_new = tree(j, 7);
            if j_new == 0
                pred(k) = tree(j, 10);
                c = size(tree,1) + 1;
            else
                j = j_new;
            end
        end
        
    end
end


end