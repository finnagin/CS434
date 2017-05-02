function [feature, theta, info_gain, l_X, l_Y, r_X, r_Y, l_flag, r_flag] = stump(X, Y)
%% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % Y - A vector containing the class to be predicted for each sample
    
%% Output:
    % feature - The feature to split the data on
    % theta - value to split the data based on the given feature
    % l_X - the left branch feature data
    % l_Y - the left branch class data
    % l_X - the right branch feature data
    % l_X - the right branch class data
    % l_flag - bollian denoting if the left branch is a leaf
    % r_flag - bollian denoting if the right branch is a leaf

                
%% code:
%      [1 features |  2 split value   |   3 info_gain    ]
info = [1:size(X,2);zeros(1,size(X,2));zeros(1,size(X,2))]';

for n = 1:size(X,2) % iterate over the features
    gain = -100; % set gain to a low number so it is easy to detect errors
    X_sorted = sortrows([X Y],n); % sort the rows based on the current feature
    Y_sorted = X_sorted(:,end); % put sorted data into Y
    X_sorted = X_sorted(:,1:(end-1)); % put sorted data into X
    split = [0 -100]; % initilize split vector so errors are easy to spot
    
    for c = 2:size(X,1) % iterate over samples
        Y_old = Y_sorted(c-1); % store last clas value
        Y_new = Y_sorted(c); % store new clas value
        if Y_old ~= Y_new % detect class change
            p1 = sum(Y(1:c-1) == 1)/(c-1); % calculate ratio of left split
            p2 = sum(Y(c:length(Y)) == 1)/(length(Y)-c+1); % calculate ratio of right split
            p = sum(Y == 1)/length(Y); % calculate original ratio
            gain_c = -p*mylog2(p) - (1-p)*mylog2(1-p) -...
                        (-(c-1)/length(Y)*(p1*mylog2(p1)+(1-p1)*mylog2(1-p1)) +...
                        -(1-(c-1)/length(Y))*(p2*mylog2(p2)+(1-p2)*mylog2(1-p2))); % calculate information gain of split
            if gain_c > gain % check if this is the highest so far
                gain = gain_c; % put into overall best gain variable
                split = [(X_sorted(c,n)+X_sorted(c-1,n))/2 gain_c]; % split between feature values and store gain
            end
        end
    end
    info(n,2:3) = split; % store best split vector
end

info = sortrows(info,3); % sort by information gain
info_gain = info(1,3); % return best information gain
theta = info(1,2); % return theta of best feature
feature = info(1,1); % return best feature
split_ind = X(:,feature) < theta; % index on theta for the best feature
l_X = X(split_ind,:); % split data into left and right
r_X = X(~split_ind,:);
l_Y = Y(split_ind);
r_Y = Y(~split_ind);

% flag for left leaf
if sum(l_Y == 1) == length(l_Y) || sum(l_Y == 1) == 0
    l_flag = true;
else
    l_flag = false;
end

% flag for right leaf
if sum(r_Y == 1) == length(r_Y) || sum(r_Y == 1) == 0
    r_flag = true;
else
    r_flag = false;
end

end
