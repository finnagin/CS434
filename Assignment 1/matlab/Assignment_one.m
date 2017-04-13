clc; close all; clear all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Trdata = load('housing_train.txt'); % load training data
Tedata = load('housing_test.txt');  % load texting data

%% Problem One
mtr = size(Trdata, 1); % obtain the size of training data
one_tr = ones(mtr, 1); % construct a vector that its entries are one

Xtr = [one_tr Trdata(:, 1:13)]; % create a matrix that contain all features
Ytr = Trdata(:, 14); % Median value of owner occupied homes in thousands

mte = size(Tedata, 1); % obtain the size of testing data
one_te = ones(mte, 1); % construct a vector that its entries are one

Xte = [one_te Tedata(:, 1:13)]; % create a matrix that contain all features
Yte = Tedata(:, 14); % Median value of owner occupied homes in thousands

%% Problem Wwo And Three
% compute optimal weight and sum of square errors 
% for training and testing data
[Tr_error, Te_error, Tr_w] = sse(Xtr, Ytr, Xte, Yte, 0);

%% Problem Four
% compute the sum of square error without introducing X 
% this says X just contain the first 13 columns of each data set

nXtr = Trdata(:, 1:13); % new X for training data
nXte = Tedata(:, 1:13); % new X for testing data

% new weight for training data and new sum of 
% square error for training and testing data
[nTr_error, nTe_error, nTr_w] = sse(nXtr, Ytr, nXte, Yte, 0);

%% Problem Six
% regularization constants
lambda = [0.01, 0.05, 0.1, 1i^(1i), 0.5, 1, 1.5, exp(1), pi, 5]; 
l = length(lambda); % obtain the of number of regularization constants

% obtain the sum of square error with different lambda
% for training data and testing data
ltr_error = zeros(l, 1); 
lte_error = zeros(l, 1);

ltrw_norm = [];
for k = 1:l
    [ltr_error(k), lte_error(k), ltr_w] = sse(Xtr, Ytr, Xte, Yte, lambda(k));
    ltrw_norm = [ltrw_norm, ltr_w'*ltr_w];
end

figure
subplot(2, 1, 1);
plot(lambda, ltr_error, 'ro');
title('SSE with Regularization For Training Data');
subplot(2, 1, 2);
plot(lambda, lte_error, 'bo');
title('SSE with Regularization For Testing Data');
hold off

% When we increase lambda, the sum of square error will get larger
% since the model we have is underfitting. Thus, by adding more
% regularization can not help to fix the problem.

%% Problem Five and Seven
Xfr = Xtr; % rename training data set, then add some random features 
Xfe = Xte; % rename testing data set, then add some random features 
fr_error = []; % obtain sse when we add features on training data
fe_error = []; % obtain sse when we add features on testing data

% obtain weights when we add features 
% with different regularization 
% constant for training data
train_wlfr = []; 

% obtain the 2-norm of optimal weight vector
% for problem eight
w_norm = [];

% obtain sum of square error when adding additional random features with
% different lambda
lfe_error = [];

rfeatures = [1 5 4 12 6 10 2 3 9 8 11 7 10]; % random features

for f = rfeatures
    % uniform features distribute on [0,a]
    
    % add uniform features 
    Xfr = [(f+1)*rand(mtr, 1) f*rand(mtr, 1) Xfr];
    Xfe = [(f+1)*rand(mte, 1) f*rand(mte, 1) Xfe];
    
    [r_error, e_error, wfr] = sse(Xfr, Ytr, Xfe, Yte, 0);
    w_norm = [w_norm, wfr'*wfr];
    
    % r_error is training error, e_error is testing error 
    % and wfr is weight for training data
    fr_error = [fr_error; r_error];
    fe_error = [fe_error; e_error];
    
    % compute weight with regularization constants 
    % on training data and tesing data
    for k = 1:l
        [lr_error, le_error, wlfr] = sse(Xfr, Ytr, Xfe, Yte, lambda(k)); 
    end
    
    train_wlfr = [train_wlfr; {wlfr}];
    lfe_error = [lfe_error; lr_error, le_error];

end

figure
subplot(2, 1, 1);
plot(fr_error, 'ro');
title('SSE with Additional Random Features For Training Data');
subplot(2, 1, 2);
plot(fe_error, 'bo')
title('SSE with Additional Random Features For Testing Data');
hold off

% The model(linear regression with order 1) we have is 
% underfitting with huge error. When we add more features to the model,
% we are adding more data points to the model then the optimal 
% weight to some features will either increase or decrease. Some slightly
% or un-useful feature will has lower bias(weight), some meaningful 
% features will have higher bias(weight).
% 
% However, since we are adding meaningless features to the model, they
% won't help with the testing model. Our testing data doesn't have any
% meaningless features, the model we obtain from the training data won't
% help us to predict the expected values. Thus, the sum of square errors
% will increase when we keep adding features.

%% Addition Test For Problem Seven
% compute the mean of sum of square error of training data
mfr_error = zeros(13, 1); 
% compute the mean of sum of square error of testing data
mfe_error = zeros(13, 1); 

fel = length(rfeatures); % obtain the number of features

for n = 1:fel
    [mfr_er, mfe_er, mwftr] = fsse(Xtr, Ytr, Xte, Yte, rfeatures, 0);
    mfr_error = mfr_error + mfr_er;
    mfe_error = mfe_error + mfe_er;
end

% mean sse of training data with additional features
result1 = mfr_error*(1/fel); 
% mean sse of tresting data with additional features
result2 = mfe_error*(1/fel);

figure
subplot(2, 1, 1);
plot(result1, 'ro');
title('Mean SSE with Additional Random Features For Training Data');
subplot(2, 1, 2);
plot(result2, 'bo')
title('Mean SSE with Additional Random Features For Testing Data');
hold off

%% Problem Eight
reg = lambda'*w_norm; % compute the regularization with different lambda
[reg_m, reg_n] = size(reg); % obtain the size of regularization term

% compute training error with regularization term
tr_er = ones(reg_m, 1)*fr_error' + reg;
% compute testing error with regularization term
te_er = ones(reg_m, 1)*fe_error' + reg;

figure
subplot(2, 1, 1)
plot(tr_er(1,:), 'ro');
title(['SSE With Regularization Term When \lambda = ', num2str(lambda(1))])
subplot(2, 1, 2)
plot(te_er(1,:), 'bo');
title(['SSE With Regularization Term When \lambda = ', num2str(lambda(1))])
hold off

% Yes, we can. From the resulting graph, we obtain two similar graphs with
% same behavior when we compute the optimal weight by using 
% $w = (X^TX + \lambda I)^{-1}X^TY$.

figure
plot(lambda, ltrw_norm, 'o');
title(['||w|| With \lambda '])
hold off