function [fr_error, fe_error, wftr] = fsse(Xfr, Ytr, Xfe, Yte, features, lambda)
% This .m file is to calculate the sum of square error
% when we add features to the training data set

mtr = size(Xfr, 1); % obtain the size of training data
mte = size(Xfe, 1); % obtain the size of testing data

fr_error = []; % save sum of square error of training data with features
fe_error = []; % save sum of square error of testing data with features
wftr =[]; % obtain the optimal weight

for f = features
    % add uniform features 
    Xfr = [(f+1)*rand(mtr, 1) f*rand(mtr, 1) Xfr];
    Xfe = [(f+1)*rand(mte, 1) f*rand(mte, 1) Xfe];
    
    [r_error, e_error, wfr] = sse(Xfr, Ytr, Xfe, Yte, lambda);
    wftr = [wftr, {wfr}];
    
    % r_error is training error, e_error is testing error 
    % and wfr is weight for training data
    fr_error = [fr_error; r_error];
    fe_error = [fe_error; e_error];
    
end

end