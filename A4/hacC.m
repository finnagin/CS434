function [Kgroups, tree] = hacC(X, Layers, K)
%% Parameters:
    % Xr - An array containing the training feature vectors from each of the samples
    % Yr - A vector containing the training class to be predicted for each sample
    % Xe - An array containing the feature vectors from each of the samples
    %      for prediction
    % K - the number of neighbors to consider

%% code:

samples = size(X,1);
groups = 1:samples;
tree = zeros(Layers-1,3);
itr = 1;

h = waitbar(0,'Running hacC...');
while length(unique(groups)) > 1
    distance = [0 0 inf];
    for b = unique(groups)
        testD = [0 0 inf];
        test1 = find(groups == b);
        for c = unique(groups(groups ~= b))
            testD2 = [0 0 0];
            test2 = find(groups == c);
            for d1 = test1
                for d2 = test2
                    X1 = X(d1,:);
                    X2 = X(d2,:);
                    if norm(X1 - X2) > testD2(end)
                        testD2 = [b c norm(X1 - X2)];
                    end
                end
            end
            if testD2(end) < testD(end)
                testD = testD2;
            end
        end
        if testD(end) < distance(end)
            distance = testD;
        end
    end
    replace = (groups == distance(2));
    groups(replace) = repmat(distance(1),1,sum(replace));
    if length(unique(groups)) < Layers
        tree(itr,:) = distance;
        itr = itr+1;
    end
    waitbar(1 - length(unique(groups))/length(groups))
    if length(unique(groups)) == K
        Kgroups = groups;
    end
end

close(h)
    
end