function pred = treeRead(X_Test, tree)
%% Parameters:
    % X_Test - The feature data for testing
    % tree - the decision tree
                
                
%% tree cheat sheet:         
% tree - | 1 feature | 2 theta | 3 info gain | 4 left flag | 5 right flag |
% 6 left child | 7 right child | 8 parent | 9 left prediction | 10 right prediction |

%% Output:
% pred - a vector of predictions based on feature data

%% Code:

pred = zeros(1, size(X_Test,1))'; % initialize prediction vector to # of samples

for k = 1:size(X_Test,1) % iterate over all samples
    c = 0; % initialize tree node counter
    node = 1; % initilize feature
    while c <= size(tree,1) % iterate over nodes
        c = c+1; % add to node counter
        if X_Test(k,tree(node,1)) < tree(node,2) % check if left brance
            node_new = tree(node, 6); % set next node
            if node_new == 0 || tree(node,4) % check for end of tree
                pred(k) = tree(node, 9); % make prediction
                c = size(tree,1) + 1; % end while loop
            else
                node = node_new; % set node for next loop
            end
            
        end
        
        if X_Test(k,tree(node,1)) >= tree(node,2) % check if left brance
            node_new = tree(node, 7); % set next node
            if node_new == 0 || tree(node,5) % check for end of tree
                pred(k) = tree(node, 10); % make prediction
                c = size(tree,1) + 1; % end while loop
            else
                node = node_new; % set node for next loop
            end
        end
        
    end
end


end