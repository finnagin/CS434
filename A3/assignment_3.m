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

Y_Train = Train(:,1); % load the class data into a vector
Y_Test = Test(:,1); % load the class data into a vector

X_Train = Train(:,2:end); % load the feature data into an array
X_Test = Test(:,2:end); % load the feature data into an array


%% Normalize data

for n = 1:size(X_Train,2) % loop through each feature
    X_Train(:,n) = X_Train(:,n) - min([X_Train(:,n) ; X_Test(:,n)]); % shift data so that its min is zero
    X_Train(:,n) = X_Train(:,n) ./ max([X_Train(:,n) ; X_Test(:,n)]); % divide by the max so that its range is [0,1]
end

for n = 1:size(X_Test,2) % loop through each feature
    X_Test(:,n) = X_Test(:,n) - min([X_Train(:,n) ; X_Test(:,n)]); % shift data so that its min is zero
    X_Test(:,n) = X_Test(:,n) ./ max([X_Train(:,n) ; X_Test(:,n)]); % divide by the max so that its range is [0,1]
end

%%
for n = 1:size(X_Train,2) % loop through each feature
    X_Train(:,n) = (X_Train(:,n) - mean([X_Train(:,n) ; X_Test(:,n)]))./std([X_Train(:,n); X_Test(:,n)]); 
    X_Test(:,n) = (X_Test(:,n) - mean([X_Train(:,n) ; X_Test(:,n)]))./std([X_Train(:,n); X_Test(:,n)]);
end


%% Problem 1.2

K = 1:2:109; % set up a vector of K values
Train_error = zeros(1,size(K,2)); % initialize a vector for training error
eps = zeros(1,size(K,2)); % initialize a vector for cross validation error
Test_error = zeros(1,size(K,2)); % initialize a vector for testing error

for n = 1:size(K,2) % loop through all K values
    P_Train = knnPred(X_Train, Y_Train, X_Train, K(n)); % predict for training data
    Train_error(n) = sum(P_Train ~= Y_Train)/size(Y_Train,1); % sum errors and normalize to [0,1]
    for m = 1:size(X_Train, 1) % loop through each sample to leave out
        X2 = X_Train(1:size(X_Train,1)~=m,:); % create training X leaving one out
        Y2 = Y_Train(1:size(Y_Train,1)~=m,:); % create training Y leaving one out
        P2 = knnPred(X2, Y2, X_Train(m,:), K(n)); % predict on the sample left out
        eps(n) = (eps(n) + (P2 ~= Y_Train(m))); % add errors to cross validation error
    end
    eps(n) = eps(n)/m; % average cross validation error
    P_Test = knnPred(X_Train, Y_Train, X_Test, K(n)); % predict for testing data
    Test_error(n) = sum(P_Test ~= Y_Test)/size(Y_Test,1); % sum errors and normalize to [0,1]
end

% plot the 3 errors (testing and training were normalized so that they 
% would be on the same scale as the cross validation error)

figure
hold on
plot(K, Train_error, 'bo-')
plot(K, eps, 'go-')
plot(K, Test_error, 'mo-')
legend('Training error', 'Epsilon', 'Testing Error','Location','SouthEast')
xlabel('K')
ylabel('Percent missed')
ylim([0 0.15])
hold off


%% Unnormalize data

Y_Train = Train(:,1); % load the class data into a vector
Y_Test = Test(:,1); % load the class data into a vector

X_Train = Train(:,2:end); % load the feature data into an array
X_Test = Test(:,2:end); % load the feature data into an array


%% 2.1

% The decision stump function is defined in the stump.m file

Stump_1 = treePlanter(X_Train, Y_Train, 1);
pred_Test = treeRead(X_Test, Stump_1);
pred_Train = treeRead(X_Train, Stump_1);
test_error = sum(Y_Test ~= pred_Test)
train_error = sum(Y_Train ~= pred_Train)

%% 2.2




l = 1:10;
test_acc = zeros(1,length(l));
train_acc = zeros(1,length(l));

for d = 1:length(l)
    tree = treePlanter(X_Train, Y_Train, l(d));
    pred = treeRead(X_Test, tree);
    test_acc(d) = sum(Y_Test ~= pred)/length(Y_Test);
    pred = treeRead(X_Train, tree);
    train_acc(d) = sum(Y_Train ~= pred)/length(Y_Train);
end

figure
plot(l, test_acc, 'bo-', l , train_acc, 'mo-')

%%

tree = treePlanter(X_Train, Y_Train, 7);
used_Features = unique(sort(tree(:,1)));
