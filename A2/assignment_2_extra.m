%% Assignment 2
% Finn Womack & Rong Yu

% This clears the data from the following run
clc; 
close all; 
clear all;

% this formats the way the numbers on the output are displayed
format long g

%% Loading data

Train = load('usps-4-9-train.csv'); % load training data
Test = load('usps-4-9-test.csv');  % load texting data

%% Create Variables

% note for Y: 0 for 4 ; 1 for 9
% note for X&Y: NxM = samples x features

Xtrain = Train(:, 1:(end-1)); % load the training featur
Ytrain = Train(:, end); % load the training classes

Xtest = Test(:, 1:(end-1)); % load the testing features
Ytest = Test(:, end); % load the testing classes

%% Problem 1
% the jittery learning rates were removed from the graph since they obscured the best
% learning rate.
learnRates = [2e-6 1e-6 3e-7 2e-7 1e-7]; % set a vector of learining rates
N = 1500; % set the number of iterations
losses = zeros(size(learnRates,2),N); % initialize the loss array
% for loss array: NxM = learn rate x iteration

figure % open new figure
hold on
for k = 1:size(learnRates,2) % loop through each learnrate
    % run the batch gradient decent algorithem with the loss flag set to 1:
    % (This has the function output a vector of the losses for each
    % iteration)
    losses(k,:) = gradBatch(Xtrain, Ytrain, N, learnRates(k), 0, 1);
    plot(1:N, losses(k,:), '-') % plot the result
end

ylim([0 50]) % set the max loss value so the figure isn't too small to see
legend(strread(num2str(learnRates),'%s')) % generate the legend
hold off

%% Problem 2

% initialize the training and testing accuracy vectors
ATrain = zeros(N, 1);
ATest = ATrain;

W = gradVec(Xtrain, Ytrain, 1500, 2e-6, 0); % calculate W

for k = 1:N
    %W = gradBatch(Xtrain, Ytrain, iterations(k), learner, 0);
    
    % Compute training accuracy
    Pred = (1./(1+exp(-W(k,:)*Xtrain')) >= 1/2); % generate Prediction on training data
    error = sum(abs(Ytrain - Pred')); % sum the errors
    ATrain(k) = 1 - error/size(Xtrain,1); % put success rate into accuracy vector

    % Compute testing accuracy
    Pred = (1./(1+exp(-W(k,:)*Xtest')) >= 1/2); % Prediction on testing data
    error = sum(abs(Ytest - Pred')); % sum the errors
    ATest(k) = 1 - error/size(Xtest,1); % put success rate into accuracy vector
    
end

% plot the accuracies vs the iterations
figure
plot(1:1500, ATrain, 'r-', 1:1500, ATest, 'b-')
xlabel('Iterations')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
ylim([.9 1])
hold off

W = gradVec(Xtrain, Ytrain, 1500, 1e-7, 0); % calculate W

for k = 1:N
    %W = gradBatch(Xtrain, Ytrain, iterations(k), learner, 0);
    
    % Compute training accuracy
    Pred = (1./(1+exp(-W(k,:)*Xtrain')) >= 1/2); % generate Prediction on training data
    error = sum(abs(Ytrain - Pred')); % sum the errors
    ATrain(k) = 1 - error/size(Xtrain,1); % put success rate into accuracy vector

    % Compute testing accuracy
    Pred = (1./(1+exp(-W(k,:)*Xtest')) >= 1/2); % Prediction on testing data
    error = sum(abs(Ytest - Pred')); % sum the errors
    ATest(k) = 1 - error/size(Xtest,1); % put success rate into accuracy vector
    
end

% plot the accuracies vs the iterations
figure
plot(1:1500, ATrain, 'r-', 1:1500, ATest, 'b-')
xlabel('Iterations')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
ylim([.9 1])
hold off


%% Problem 3

% the sudocode for the regularization:

% Given: training examples (xi, yi), i = 1, ..., N
% W <- [0, 0, ..., 0]
% Repeat until convergence:
%   d <- [0, 0, ..., 0]
%   For i = 1 to N do:
%       \hat{yi} <- 1/(1 + e^(-w*xi))
%       error = yi - \hat{yi}
%       d = d + error * xi
%   w <- w + n * (d + lamda * w)

%% Problem 4

lambdas = arrayfun(@(x) x*10^3, 0.5:4:64.5); % generate the vector of lambdas

% initialize the training and testing accuracy vectors
ATrain = zeros(size(lambdas,2), 1);
ATest = ATrain;

for k = 1:size(lambdas,2)
    % generate W at each lambda
    W = gradBatch(Xtrain, Ytrain, 1000, 2e-6, lambdas(k)); 
    
    
    Pred = (1./(1+exp(-W*Xtrain')) >= 1/2); % generate Prediction on training data
    error = sum(abs(Ytrain - Pred')); % sum the errors
    ATrain(k) = 1 - error/size(Xtrain,1); % put success rate into accuracy vector

    
    Pred = (1./(1+exp(-W*Xtest')) >= 1/2); % generate Prediction on testing data
    error = sum(abs(Ytest - Pred')); % sum the errors
    ATest(k) = 1 - error/size(Xtest,1); % put success rate into accuracy vector
end

% generate plot of accuracies vs lambdas
figure
plot(lambdas, ATrain, 'ro-', lambdas, ATest, 'bo-')
xlabel('Lambda')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
xlim([lambdas(1) lambdas(end)])
hold off
