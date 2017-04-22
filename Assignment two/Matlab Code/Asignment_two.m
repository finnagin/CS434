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
learning_rate = [1e-7 2e-7 1e-6 2e-6];
num_lr = length(learning_rate); % obtain the number of learning rate 

% initial optimal weight
initial_w = zeros(Train_features, 1);
% iterations
iter = 800;

figure
hold on
for n = 1:num_lr
    loss = Batgrad(X_train, Y_train, iter, initial_w, learning_rate(n), 0, 1);
    plot(1:iter, loss, '-')
end
ylim([0 50])
legend(strread(num2str(learning_rate),'%s'))
hold off

best_rate = 2e-6;

%% Problem Two
Iterations = [100 150 200 250 300 350 400 450 500];
num_ite = length(Iterations);

% store training accuracy and testing accuracy
train_accuracy = zeros(num_ite, 1);
test_accuracy = train_accuracy;

for i = 1:num_ite
    train_w = Batgrad(X_train, Y_train, Iterations(i),...
        initial_w, best_rate, 0, 0);
    
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
% $$\sum_{i=1}^m l(g(w^Tx^i,y^i))x^i+\lambda*w$$
%

%% Problem Four
len = 1:0.5:7; % numbers of lambda
Lambda = 10.^len;
Num_lam = length(Lambda);

% store training accuracy and testing accuracy with regularization term
re_train_accuracy = zeros(Num_lam, 1);
re_test_accuracy = re_train_accuracy;

for k = 1:Num_lam
    re_train_w = Batgrad(X_train, Y_train, 200, initial_w,...
        best_rate, Lambda(k), 0);
    
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
plot(len, re_train_accuracy, 'ro-', len, re_test_accuracy, 'bo-')
xlabel('Lambda')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off
