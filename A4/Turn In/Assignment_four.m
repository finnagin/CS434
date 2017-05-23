%% Assignment 4
% Finn Womack & Rong Yu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Instructions:
% 1) Make sure all the .m files and data are in the same folder
% 2) open assignment_four.m in matlab
% 3) Click the run button at the top.
% 4) You will most likely be prompted to change the current folder. 
%    Press the add to path button.
% 5) The code will then run and display the relevant graphs in new windows. 
%    The relevant calculations will be displayed in the bottom comand 
%    window. You will likely have to scroll up to see all of them.

%%

% This clears the data from the previous run
clc; close all; clear all;

% this formats the way the numbers on the output are displayed
format long g

% set the rng seed so the random results are the same each run
rng(112358)

%% Load Data
X = load('data-2.txt');

%% K-means Problem One
K = 2;
[group, sse]= Kmeans(X, K);
iter = length(sse);

figure
plot(1:iter, sse, '-o')
xlabel('iteration')
ylabel('sse')
title(['K-Means Clusering with k = ', num2str(K)])

%% Read Image
% this was just an extra thing we did
% it displays a sample of each cluster
% it could make a larger display when there were more samples in the data

for k = 1:K % iterate through each cluster
    figure % make figure
    image = ReadImage(group, X, k, 1); % display image
end

%% K-means Problem Two
K2 = 2:10; % vector of k values
L = length(K2); % # of k values
iteration = 10; % # of iterations
SSE = zeros(L, 1); % initialize sse vector

for k = 1:L
    minerr = zeros(iteration, 1); % initialize error vector
    for itr = 1:iteration % iterate chosen number of times
        [G, S] = Kmeans(X, K2(k)); % run kmeans function
        minerr(itr) = S(end); % store sse
    end
    SSE(k) = min(minerr); %find minimum sse for this iteration
end

% display the figure
figure
plot(K2, SSE, '-o')
xlabel('K')
ylabel('SSE')
title('SSE with K = 2 to 10')

%% Single Link (Problem 3)
K = 10; % set # of clusters

[groupS, clustS, infoS] = SingleL(X, K); % run the single link function
clustS = cell2mat(clustS); % convert cluster to a matrix
clustS = [clustS; infoS(2,1) infoS(3,1), infoS(3,2)]; % format cluster for diagram

% This ranks the cluster indexes from small to large
[~,~,ranks]=unique(clustS(:,1:2));
ranks = [ranks(1:(length(ranks)/2)) ranks((length(ranks)/2+1):end)];
clustS(:,1:2) = ranks;

% this displays the figure
figure
dendrogram(clustS)

%% Complete Link (Problem 4)
[groupC, clustC, infoC] = CompleteL(X, K);% run the complete link function
clustC = cell2mat(clustC); % convert cluster to a matrix
clustC = [clustC; infoC(2,1) infoC(3,1), infoC(3,2)]; % format cluster for diagram

% This ranks the cluster indexes from small to large
[~,~,ranks]=unique(clustC(:,1:2));
ranks = [ranks(1:(length(ranks)/2)) ranks((length(ranks)/2+1):end)];
clustC(:,1:2) = ranks;

% this displays the figure
figure
dendrogram(clustC)



