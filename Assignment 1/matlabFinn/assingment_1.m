clc; 
close all; 
clear all;

format long g

%% Loading data

Train = load('housing_train.txt'); % load training data
Test = load('housing_test.txt');  % load texting data

%% Problem 1

Xtrain = [ones(size(Train,1), 1) Train(:, 1:13)];
Ytrain = Train(:, 14);

Xtest = [ones(size(Test,1), 1) Test(:, 1:13)];
Ytest = Test(:, 14);

%% Problem 2

W = solveW(Xtrain,Ytrain)

%% Problem 3

SSEtrain = sseSolve(Xtrain,Ytrain,W)

SSEtest = sseSolve(Xtest,Ytest,W)

%% Problem 4

Xtrain2 = Train(:, 1:13);
Ytrain2 = Train(:, 14);

Xtest2 = Test(:, 1:13);
Ytest2 = Test(:, 14);

W2 = solveW(Xtrain2,Ytrain2)

SSEtrain2 = sseSolve(Xtrain2,Ytrain2,W2)

SSEtest2 = sseSolve(Xtest2,Ytest2,W2)

%% Problem 5

rng(1123581321)


%maximums = [1 5 4 12 6 10 2 3 9 8 11 7];
maximums = randperm(12);

SSEavg = zeros(2, 12);

for k = 1:100 
    SSEmat = cell2mat(arrayfun(@(z) randomSolve(Xtrain, Ytrain, Xtest, Ytest, z, maximums), 1:12, 'un', 0));
    SSEavg = (SSEavg*(k-1) + SSEmat)*(1/k);
end

figure
subplot(2, 1, 1);
plot(SSEavg(1,:), 'ro-');
title('Mean SSE with Additional Random Features For Training Data');
subplot(2, 1, 2);
plot(SSEavg(2,:), 'bo-')
title('Mean SSE with Additional Random Features For Testing Data');
hold off

%% Problem 6

Lambdas = [0.01, 0.05, 0.1, 1i^(1i), 0.5, 1, 1.5, exp(1), pi, 5];

Wlambda = cell2mat(arrayfun(@(z) regW(Xtrain,Ytrain,z),Lambdas, 'un', 0));

LambdaTest = zeros(1,size(Lambdas,2));
LambdaTrain = LambdaTest;

for k = 1:size(Lambdas,2)
    LambdaTrain(k) = sseSolve(Xtrain, Ytrain, Wlambda(:,k));
    LambdaTest(k) = sseSolve(Xtest, Ytest, Wlambda(:,k));
end

figure
subplot(2, 1, 1);
plot(Lambdas, LambdaTrain, 'ro-');
title('SSE with Regularization For Training Data');
subplot(2, 1, 2);
plot(Lambdas, LambdaTest, 'bo-');
title('SSE with Regularization For Testing Data');
hold off

%% Problem 7

WNorms = zeros(1,size(Lambdas,2));

for k = 1:size(Lambdas,2)
    WNorms(k) = norm(Wlambda(:,k));
end

figure
plot(Lambdas, WNorms, 'mo-');
title('W Norms with Regularization For Training Data');
hold off

%% Display 