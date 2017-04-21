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
%learnRates = arrayfun(@(x) (10)^-x, 6:9);
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

learner = 1e-7; % set learning rate to use
%iterations =[1 2:2:1500];

% initialize the training and testing accuracy vectors
ATrain = zeros(N, 1);
ATest = ATrain;

W = gradVec(Xtrain, Ytrain, 1500, learner, 0); % calculate W

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

%% Problem 4

lambdas = arrayfun(@(x) 10^x, 3.5:.25:5.5); % generate the vector of lambdas
%lambdas = [1 10 30 50 70 100].*10^2;

% initialize the training and testing accuracy vectors
ATrain = zeros(size(lambdas,2), 1);
ATest = ATrain;

for k = 1:size(lambdas,2)
    % generate W at each lambda
    W = gradBatch(Xtrain, Ytrain, 1000, learner, lambdas(k)); 
    
    
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
xlabel('Lambda (10^x)')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off