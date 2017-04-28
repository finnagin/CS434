%% Assignment 3
% Finn Womack & Rong Yu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Instructions:
% 1) Make sure all the .m files and data are in the same folder
% 2) open assignment_3.m in matlab
% 3) Click the run button at the top.
% 4) You will most likely be prompted to change the current folder. 
%    Press the add to path button.
% 5) The code will then run and display the relevant graphs in new windows. 
%    The relevant calculations will be displayed in the bottom comand 
%    window. You will likely have to scroll up to see all of them.

%%

% This clears the data from the following run
clc; 
close all; 
clear all;

% this formats the way the numbers on the output are displayed
format long g

%% Loading data

Train = load('knn_train.csv'); % load training data
Test = load('knn_test.csv');  % load texting data

%% Split independent and dependent variables

Y_Train = Train(:,1);
Y_Test = Test(:,1);

X_Train = Train(:,2:end);
X_Test = Test(:,2:end);

%% Normalize data


%% Problem 1.2

K = 1:2:59;
Train_error = zeros(1,size(K,2));
eps = zeros(1,size(K,2));
Test_error = zeros(1,size(K,2));

for n = 1:size(K,2)
    P_Train = knnPred(X_Train, Y_Train, X_Train, K(n));
    Train_error(n) = sum(P_Train ~= Y_Train)/size(Y_Train,1);
    for m = 1:size(X_Train, 1)
        X2 = X_Train(1:size(X_Train,1)~=m,:);
        Y2 = Y_Train(1:size(Y_Train,1)~=m,:);
        P2 = knnPred(X2, Y2, X_Train(m,:), K(n));
        eps(n) = (eps(n) + (P2 ~= Y_Train(m)));
    end
    eps(n) = eps(n)/m;
    P_Test = knnPred(X_Train, Y_Train, X_Test, K(n));
    Test_error(n) = sum(P_Test ~= Y_Test)/size(Y_Test,1);
end

figure
hold on
plot(K, Train_error, 'bo-')
plot(K, eps, 'go-')
plot(K, Test_error, 'mo-')
legend('Training error', 'Epsilon', 'Testing Error','Location','SouthEast')
xlabel('K')
ylabel('Percent missed')
hold off

%% 2.1

