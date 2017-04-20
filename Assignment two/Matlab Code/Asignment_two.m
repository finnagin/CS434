clc; clear all; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Train_data = load('usps-4-9-train.csv'); % load training data
Test_data = load('usps-4-9-test.csv'); % load testing data

[Train_samples, Train_features] = size(Train_data); % size of training data
[Test_samples, Test_features] = size(Test_data); % size of testing data

% Obtain all features of training data
X_train = [ones(Train_samples, 1) Train_data(:, 1:256)];
Y_train = Train_data(:, 257);

% Obtain all features of testing data
X_test = [ones(Test_samples, 1) Test_data(:, 1:256)]; 
Y_test = Test_data(:, 257); 

%% Problem One
learning_rate = (7:0.001:7.18)*1e-8;
num_lr = length(learning_rate); % obtain the number of learning rate 

% obtain the norm of each w and loss with different learning rate
w_norm = zeros(length(learning_rate), 1); 
loss = w_norm;

% initial optimal weight
initial_w = zeros(size(X_train, 2), 1);

% initialize the first loss
pre_loss = 0;

for r = 1:num_lr
    w = Batgrad(X_train, Y_train, 100, initial_w, learning_rate(r), 0);
    w_norm(r) = sqrt(w'*w);
    loss(r) = LossFunc(X_train, Y_train, w);
    if abs(loss(r) - pre_loss) < 1e-3 
        break
    end
    pre_loss = loss(r);
end

figure
plot(learning_rate, loss, 'o-');
title('Batch Gradient Decent with Different Learning Rate')
xlabel('Learning Rate')
ylabel('Loss')
hold off

% r = 7.16e-8 is our the desire learning rate. 
% If we pick r greater than 7.16e-8, our prediction will always one; 
% if we pick r somewhat less than 7.16e-8, 
% our prediction will always or get close to zero. In each case, our loss 
% will be not a number (NaN) or close to positive/negative infinity

%% Problem Two
Iterations = [100 200 300 400 500];
num_ite = length(Iterations);

% store training accuracy and testing accuracy
train_accuracy = zeros(num_ite, 1);
test_accuracy = train_accuracy;

for i = 1:num_ite
    train_w = Batgrad(X_train, Y_train, Iterations(i), initial_w, 7.16e-8, 0);
    
    % Compute training accuracy
    train_pre = logclassify(sigmoid(X_train*train_w)); % Prediction on training data
    train_error = sum(abs(Y_train - train_pre));
    train_accuracy(i) = 1 - train_error/Train_samples;

    % Compute testing accuracy
    test_pre = logclassify(sigmoid(X_test*train_w)); % Prediction on testing data
    test_error = sum(abs(Y_test - test_pre));
    test_accuracy(i) = 1 - test_error/Test_samples;
end

figure
plot(Iterations, train_accuracy, 'ro-', Iterations, test_accuracy, 'bo-')
xlabel('Iterations')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off

% When we increace the number of iterations, the accuracy of training data
% increase; however, the testing accuracy decrease.

%% Problem Three
%
% If we differentiate the loss function respect to w, we have
% $$\sum_{i=1}^m l(g(w^Tx^i,y^i))x^i+\lambda||w||_2$$
%

%% Problem Four
Lambda = [10e-3 10e-2 10e-1 1 10e1 10e2 10e3];
Num_lam = length(Lambda);

% store training accuracy and testing accuracy with regularization term
re_train_accuracy = zeros(Num_lam, 1);
re_test_accuracy = re_train_accuracy;

for k = 1:Num_lam
    re_train_w = Batgrad(X_train, Y_train, 200, initial_w, 7.16e-8, Lambda(k));
    
    % Compute training accuracy
    re_train_pre = logclassify(sigmoid(X_train*re_train_w)); % Prediction on training data
    re_train_error = sum(abs(Y_train - re_train_pre));
    re_train_accuracy(k) = 1 - re_train_error/Train_samples;

    % Compute testing accuracy
    re_test_pre = logclassify(sigmoid(X_test*re_train_w)); % Prediction on testing data
    re_test_error = sum(abs(Y_test - re_test_pre));
    re_test_accuracy(k) = 1 - re_test_error/Test_samples;
end

figure
plot(Lambda, re_train_accuracy, 'ro-', Lambda, re_test_accuracy, 'bo-')
xlabel('Lambda')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off

% If we puted small enough regularization to the optimal weight, we would
% obtain good accuracy; if too much regularization, the accuracy will drop
% since we tend to predict one more than to predict zero.
