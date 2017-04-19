%% Assignment 2
% Finn Womack & Rong Yu

clc; 
close all; 
clear all;

format long g

rng(112358)

%% Loading data

Train = load('usps-4-9-train.csv'); % load training data
Test = load('usps-4-9-test.csv');  % load texting data

%% Create Variables

% 0 for 4 ; 1 for 9
Xtrain = Train(:, 1:(end-1));
Ytrain = Train(:, end);

Xtest = Test(:, 1:(end-1));
Ytest = Test(:, end);

%% Problem 1
learnRates = arrayfun(@(x) (x*10)^-8, .5:.5:4);
N = 50;
BatSize = 200;
norms = zeros(1,size(learnRates,2));
losses = norms;

for k = 1:size(learnRates,2)
    learn = learnRates(k);
    W = gradBatch(Xtrain, Ytrain, N, learn, BatSize, 0);
    norms(k) = sqrt(W*W');
    losses(k) = -Ytrain'*log(1./(1+exp(-W*Xtrain')))' - (1 - Ytrain')*log(1 - 1./(1+exp(-W*Xtrain')))';
end

plot(.5:.5:4,losses, 'o-')

%% Problem 2

learner = 10^-8;
iterations = [10 50 100 200 300 400 500];
losses = zeros(1,size(iterations,2));

ATrain = zeros(size(iterations,2), 1);
ATest = ATrain;


for k = 1:size(iterations,2)
    W = gradBatch(Xtrain, Ytrain, iterations(k), learner, BatSize, 0);
    
    % Compute training accuracy
    Pred = (1./(1+exp(-W*Xtrain')) >= 1/2); % Prediction on training data
    error = sum(abs(Ytrain - Pred'));
    ATrain(k) = 1 - error/size(Xtrain,2);

    % Compute testing accuracy
    Pred = (1./(1+exp(-W*Xtest')) >= 1/2); % Prediction on testing data
    error = sum(abs(Ytest - Pred'));
    ATest(k) = 1 - error/size(Xtest,2);
    
end

figure
plot(iterations, ATrain, 'ro-', iterations, ATest, 'bo-')
xlabel('Iterations')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off

%% Problem 3

%% Problem 4

lambdas = arrayfun(@(x) 10^-x, -3:3);


ATrain = zeros(size(lambdas,2), 1);
ATest = ATrain;

for k = 1:size(lambdas,2)
    W = gradBatch(Xtrain, Ytrain, N, learner, BatSize, lambdas(k));
    
    
    Pred = (1./(1+exp(-W*Xtrain')) >= 1/2); 
    error = sum(abs(Ytrain - Pred'));
    ATrain(k) = 1 - error/size(Xtrain,2);

    
    Pred = (1./(1+exp(-W*Xtest')) >= 1/2); 
    error = sum(abs(Ytest - Pred'));
    ATest(k) = 1 - error/size(Xtest,2);
end

figure
plot(lambdas, ATrain, 'ro-', lambdas, ATest, 'bo-')
xlabel('Lambda')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off