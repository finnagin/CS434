%% Assignment 2
% Finn Womack & Rong Yu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Instructions:
% 1) Make sure all the .m files and data are in the same folder
% 2) open assignment_2.m in matlab
% 3) Click the run button at the top.
% 4) You will most likely be prompted to change the current folder. 
%    Press the add to path button.
% 5) The code will then run and display the relevant graphs in new windows. 
%    The relevant calculations will be displayed in the bottom comand 
%    window. You will likely have to scroll up to see all of them.

%%

clc; close all; clear all;

format long g

%% Load Data
X = load('data-1.txt');

%% K-means Problem One
rng(112358)

K = 4;
[group, sse] = Kmeans(X, K);
iter = length(sse);

figure
plot(1:iter, sse, '-o')
xlabel('iteration')
ylabel('sse')
title(['K-Means Clusering with K = ', num2str(K)])

%% Read Image
for k = 1:K
    figure
    image = ReadImage(group, X, k, 30);
end

%% K-means Problem Two
K2 = 2:10;
L = length(K2);
iteration = 10;
SSE = zeros(L, 1);

h = waitbar(0,'Running Iterations...');
for k = 1:L
    minErr = zeros(iteration, 1);
    for itr = 1:iteration
        [~, S] = Kmeans(X, K2(k));
        minErr(itr) = S(end);
    end
    SSE(k) = min(minErr);
    waitbar(k/L)
end
close(h)

figure
plot(2:10, SSE, '-o')
xlabel('K')
ylabel('SSE')
title('K-Means Clusering with K = 2 to 10')

%% Problem 3


[grps, tree] = hacS(X(randperm(6000,100),:),10,15);

tree2 = tree;

[~,~,ranks]=unique(tree(:,1:2));
ranks = [ranks(1:(length(ranks)/2)) ranks((length(ranks)/2+1):end)];
tree2(:,1:2) = ranks;

figure()
dendrogram(tree2)

%% Problem 4


[grps3, tree3] = hacC(X(randperm(6000,100),:),10,15);

tree4 = tree3;

[~,~,ranks]=unique(tree3(:,1:2));
ranks = [ranks(1:(length(ranks)/2)) ranks((length(ranks)/2+1):end)];
tree4(:,1:2) = ranks;

figure()
dendrogram(tree4)