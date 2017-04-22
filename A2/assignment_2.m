%% Assignment 2
% Finn Womack & Rong Yu

clc; 
close all; 
clear all;

format long g

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
%learnRates = arrayfun(@(x) (10)^-x, 6:9);
learnRates = [2e-6 1e-6 3e-7 2e-7 1e-7];
N = 1500;
losses = zeros(size(learnRates,2),N);

figure
hold on
for k = 1:size(learnRates,2)
    learn = learnRates(k);
    losses(k,:) = gradBatch(Xtrain, Ytrain, N, learn, 0, 1);
    plot(1:N, losses(k,:), '-')
end

ylim([0 50])
legend(strread(num2str(learnRates),'%s'))
hold off

%% Problem 2

learner = 1e-7;
%iterations =[1 2:2:1500];
N = 1500;
ATrain = zeros(N, 1);
ATest = ATrain;

W = gradVec(Xtrain, Ytrain, 1500, learner, 0);

for k = 1:N
    %W = gradBatch(Xtrain, Ytrain, iterations(k), learner, 0);
    
    % Compute training accuracy
    Pred = (1./(1+exp(-W(k,:)*Xtrain')) >= 1/2); % Prediction on training data
    error = sum(abs(Ytrain - Pred'));
    ATrain(k) = 1 - error/size(Xtrain,2);

    % Compute testing accuracy
    Pred = (1./(1+exp(-W(k,:)*Xtest')) >= 1/2); % Prediction on testing data
    error = sum(abs(Ytest - Pred'));
    ATest(k) = 1 - error/size(Xtest,2);
    
end

figure
plot(1:1500, ATrain, 'r-', 1:1500, ATest, 'b-')
xlabel('Iterations')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
ylim([.9 1])
hold off

%% Problem 3

%% Problem 4

lambdas = arrayfun(@(x) 10^x, -6:2:6);


ATrain = zeros(size(lambdas,2), 1);
ATest = ATrain;

for k = 1:size(lambdas,2)
    W = gradBatch(Xtrain, Ytrain, 175, learner, lambdas(k));
    
    
    Pred = (1./(1+exp(-W*Xtrain')) >= 1/2); 
    error = sum(abs(Ytrain - Pred'));
    ATrain(k) = 1 - error/size(Xtrain,2);

    
    Pred = (1./(1+exp(-W*Xtest')) >= 1/2); 
    error = sum(abs(Ytest - Pred'));
    ATest(k) = 1 - error/size(Xtest,2);
end

figure
plot(-6:2:6, ATrain, 'ro-', -6:2:6, ATest, 'bo-')
xlabel('Lambda (10^x)')
ylabel('Accuracy')
legend('Training Accuracy', 'Testing Accuracy')
hold off