%% Assignment 4
% Finn Womack & Rong Yu

% This clears the data from the following run
clc; 
close all; 
clear all;

% this formats the way the numbers on the output are displayed
format long g

%% Loading data

data = load('data.txt');

%% look at data

imshow(reshape(data(28000,:),28,28))

%%
