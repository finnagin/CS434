clc; close all; clear all; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Train_data = load('housing_train.txt'); % load training data
Test_data = load('housing_test.txt');  % load texting data

%% Problem One
m_train = size(Train_data, 1); % obtain the size of training data
one_train = ones(m_train, 1); % construct a vector that its entries are one

X_train = [one_train Train_data(:, 1:13)]; % create a matrix that contain all features
Y_train = Train_data(:, 14); % Median value of owner occupied homes in thousands

m_test = size(Test_data, 1); % obtain the size of testing data
one_test = ones(m_test, 1); % construct a vector that its entries are one

X_test = [one_test Test_data(:, 1:13)]; % create a matrix that contain all features
Y_test = Test_data(:, 14); % Median value of owner occupied homes in thousands

%% Problem Wwo And Three
% compute optimal weight and sum of square errors 
% for training and testing data
[Train_error, Test_error, Tr_w] = sse(X_train, Y_train, X_test, Y_test, 0);

%% Problem Four
% compute the sum of square error without introducing X 
% this says X just contain the first 13 columns of each data set

New_X_train = Train_data(:, 1:13); % new X for training data
New_X_test = Test_data(:, 1:13); % new X for testing data

% new weight for training data and new sum of 
% square error for training and testing data
[New_Train_error, New_Test_error, nTr_w] = sse(New_X_train,...
    Y_train, New_X_test, Y_test, 0);

%% Problem Five
Xfr = X_train; % rename training data set, then add some random features 
Xfe = X_test; % rename testing data set, then add some random features 
feature_train_error = []; % obtain sse when we add features on training data
feature_test_error = []; % obtain sse when we add features on testing data

rng(1123581321); % seed generate
rand_features = randperm(12); % generate random features

fel = length(rand_features); % obtain the number of features
% compute the mean of sum of square error of training data
mean_fr_error = zeros(fel, 1); 
% compute the mean of sum of square error of testing data
mean_fe_error = zeros(fel, 1); 

for n = 1:100
    [mfr_er, mfe_er, mwftr] = fsse(X_train, Y_train, X_test,...
        Y_test, rand_features, 0);
    mean_fr_error = mean_fr_error + mfr_er;
    mean_fe_error = mean_fe_error + mfe_er;
end

% mean sse of training data with additional features
result1 = mean_fr_error*(1/fel); 
% mean sse of tresting data with additional features
result2 = mean_fe_error*(1/fel);

figure
subplot(2, 1, 1);
plot(result1, 'ro-');
title('Mean SSE with Additional Random Features For Training Data');
xlabel('Random Features');
ylabel('Mean SSE');
subplot(2, 1, 2);
plot(result2, 'bo-')
title('Mean SSE with Additional Random Features For Testing Data');
xlabel('Random Features');
ylabel('Mean SSE');
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

%% Problem Six
% regularization constants
lambda = [0.01, 0.05, 0.1, 1i^(1i), 0.5, 1, 1.5, exp(1), pi, 5]; 
l = length(lambda); % obtain the of number of regularization constants

% obtain the sum of square error with different lambda
% for training data and testing data
re_train_error = zeros(l, 1); 
re_test_error = zeros(l, 1);

re_train_w_norm = zeros(l, 1); % obtain norm with regularization term

for k = 1:l
    [re_train_error(k), re_test_error(k), re_train_w] = sse(X_train,...
        Y_train, X_test, Y_test, lambda(k));
    re_train_w_norm(k) = sqrt(re_train_w'*re_train_w);
end

figure
subplot(2, 1, 1);
plot(lambda, re_train_error, 'ro-');
title('SSE with Regularization For Training Data');
xlabel('Lambda');
ylabel('SSE');
subplot(2, 1, 2);
plot(lambda, re_test_error, 'bo-');
title('SSE with Regularization For Testing Data');
xlabel('Lambda');
ylabel('SSE');
hold off

% When we increase lambda, the sum of square error will get larger
% since the model we have is underfitting. Thus, by adding more
% regularization can not help to fix the problem.

%% Problem Seven

figure
plot(lambda, re_train_w_norm, 'o-');
title('||w|| With \lambda ');
xlabel('Lambda');
ylabel('||w||');
hold off

% As $\lambda$ getting larger, we observe that the norm of optimal
% weight vector decrease. This is because we spread weight to those
% additional features. 

%% Problem Eight 

% $\sum_{i=1}^n(y_i - w^TX)^2 + \lambda\Vert w\Vert_2^2$
%
% Since we are minimaizing the optimal weight vector, w,
% if we keep increasing regularization constant, values of
% w will decrease because we are spreading weight to each
% features. i.e. Some features will have lower weight; some features will
% have lower weight. Thus, the magnitude of w will decrease as lambda
% increase. 