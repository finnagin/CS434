import numpy as np
from numpy import linalg as la
from scipy import stats as st
import math
import random as rnd
import warnings as wrn

def knnPred(Xr, Yr, Xe, K):
    '''
    This function takes a training set (both features and classes) and used a knn algorithm
    to make predictions on a another set of data (just features) outputting a list of classes

    Parameters:
    Xr - An array containing the training feature vectors from each of the samples
    Yr - A vector containing the training class to be predicted for each sample
    Xe - An array containing the feature vectors from each of the samples
         for prediction
    K - the number of neighbors to consider
    '''

    P = np.zeros(Xe.shape[0])  # initialize the vector of predictions

    for a in np.arange(Xe.shape[0]):  # iterate over the testing data
        nord = np.column_stack((np.arange(Xr.shape[0]), np.zeros(Xr.shape[0])))  # initialize a vector for norms and indicies
        for b in np.arange(Xr.shape[0]):  # iterate over the training data
            nord[b,1] = la.norm(Xr[b,:] - Xe[a,:])  # calculate the distance from the point to predict on for each training data point
        nord = nord[nord[:,1].argsort()]  # sort by distance
        ord = nord[:K,0].astype(int)  # extract the K closest points
        P[a] = st.mode(Yr[ord])[0]  # make prediction

    return P

def myLog2(a):
    '''
    This is a log2 function that returns 0 instead of inf for log2(0).
    The reason for this is to fix an error when calculating 0*log2(0) in this assignment.
    '''
    if a == 0:
        return 0
    return math.log2(a)


def count1(Y):
    '''
    counts the instances of 1 in a vector
    '''
    count = 0
    for a in Y:
        if a == 1:
            count += 1
    return count

def stump(X, Y):
    '''
    Parameters:
        X - An array containing the feature vectors from each of the samples
        Y - A vector containing the class to be predicted for each sample

    Output:
        feature - The feature to split the data on
        theta - value to split the data based on the given feature
        l_X - the left branch feature data
        l_Y - the left branch class data
        l_X - the right branch feature data
        l_X - the right branch class data
        l_flag - bollian denoting if the left branch is a leaf
        r_flag - bollian denoting if the right branch is a leaf
    '''
    info = np.column_stack((np.arange(X.shape[1]),np.zeros(X.shape[1]), np.zeros(X.shape[1])))

    for n in np.arange(X.shape[1]):  # iterate over the features
        gain = -100  # set gain to a low number so it is easy to detect errors
        X_sorted = X[X[:,n].argsort()]  # sort the rows based on the current feature
        Y_sorted = Y[X[:,n].argsort()]  # sort the classes to match the above sorting
        split = [0, -100]  # initialize split vector so errors are easy to spot

        for c in np.arange(1,X.shape[0]):  #  iterate over samples
            Y_old = Y_sorted[c-1]  #  store last class value
            Y_new = Y_sorted[c]  #  store new class value
            if Y_old != Y_new:  #  detect class change
                p1 = count1(Y[0:c-1])/(c)  # calculate ratio of left split
                p2 = count1(Y[c:])/(len(Y)-c)  # calculate ratio of right split
                p = count1(Y)/len(Y)  # calculate original ratio
                gain_c = -p*myLog2(p) - (1-p)*myLog2(1-p) - (-(c)/len(Y)*(p1*myLog2(p1)+(1-p1)*myLog2(1-p1)) + -(1-(c)/len(Y))*(p2*myLog2(p2)+(1-p2)*myLog2(1-p2)))
                                                                # ^calculate info gain of split
                if gain_c > gain:  # check if this is the highest info gain so far
                    gain = gain_c  # put into overall best gain variable
                    split = [(X_sorted[c,n]+X_sorted[c-1,n])/2, gain_c]  # split between feature values and store info gain
        info[n,1] = split[0]  # store best split
        info[n,2] = split[1]  # store best info gain

    info = info[info[:,2].argsort()]  #  sort by info gain
    info_gain = info[0,2]  # return best information gain
    theta = info[0,1]  # return theta of best feature
    feature = info[0,0].astype(int)  # return best feature
    split_ind_l = X[:, feature] < theta  # split left index on theta for the best feature
    split_ind_r = X[:, feature] >= theta  # and split right
    l_X = X[split_ind_l,:]  # These next 4 store the left and right data sets for both features and classes
    r_X = X[split_ind_r,:]
    l_Y = Y[split_ind_l]
    r_Y = Y[split_ind_r]

    if len(set(l_Y)) == 1:  # checks to see if the split fully classifies the data on the left
        l_flag = 1
    else:
        l_flag = 0

    if len(set(r_Y)) == 1:  # checks to see if the split fully classifies the data on the right
        r_flag = 1
    else:
        r_flag = 0

    return {'feature': feature, 'theta': theta, 'info_gain': info_gain, 'l_X': l_X, 'l_Y': l_Y, 'r_X': r_X, 'r_Y': r_Y,
            'l_flag': l_flag, 'r_flag': r_flag}




def treePlanter(X, Y, Layers):
    '''
    %% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % Y - A vector containing the class to be predicted for each sample
    % layers - the maximum depth desired
                
                
    %% Output:            
    the output of stump looks like : | 1 feature | 2 theta | 3 info gain | 4 left flag | 5 right flag | 6 left child |
                                        7 right child | 8 parent | 9 left prediction | 10 right prediction |
    '''

    sum2n = sum(2**z for z in np.arange(Layers))  # calculates the number of layers needed (2^n-1) where n = max depth of tree
    tree = np.zeros((sum2n, 10))  # initialize the tree
    skip = 0  # set skip counter to zero
    data = np.empty((2,sum2n),dtype = object)  # initialize data object array

    data[0,0] = X  # input X into data cell
    data[1,0] = Y  # input Y into data cell

    c = 0  # % initialize the child counter
    for k in np.arange(sum2n):  # interate over all possible nodes
        if (k + skip + 1) > sum2n or c < k:  # check if all nodes are finished
            for g in np.arange(len(tree[:, 6])):  # get rid of pointers to non existent branches
                if tree[g,5] >= k:
                    tree[g,5] = 0

            for g in np.arange(len(tree[:, 7])):  # get rid of pointers to non existent branches
                if tree[g,6] >= k:
                    tree[g,6] = 0

            tree = tree[0:k,:]  # trim tree
            return tree

        output = stump(data[0,k],data[1,k])  # Find the split and save to output data
        tree[k, 0] = output['feature']
        tree[k, 1] = output['theta']
        tree[k, 2] = output['info_gain']
        l_X = output['l_X']
        l_Y = output['l_Y']
        r_X = output['r_X']
        r_Y = output['r_Y']
        tree[k, 3] = output['l_flag']
        tree[k, 4] = output['r_flag']

        if count1(l_Y) > len(l_Y)/2:  # make prediction at left node
            tree[k,8] = 1
        elif count1(l_Y) < len(l_Y)/2:
            tree[k,8] = -1
        elif count1(l_Y) == len(l_Y)/2:
            tree[k, 8] = rnd.sample([-1, 1], 1)[0]
            wrn.warn('Warning: even split')

        if count1(r_Y) > len(r_Y)/2:  # make prediction at right node
            tree[k,9] = 1
        elif count1(r_Y) < len(r_Y)/2:
            tree[k,9] = -1
        elif count1(r_Y) == len(r_Y)/2:
            tree[k, 9] = rnd.sample([-1, 1], 1)[0]
            wrn.warn('Warning: even split')

        if tree[k,4]:  # check leaf flag
            skip += 1  # add to skip counter
        else:
            c += 1  # count child
            if c + 1 <= sum2n - skip:  # make sure we don't have too many nodes
                data[0,c] = l_X  # store child data
                data[1,c] = l_Y
                tree[k,5] = c  # store child pointer
                tree[c,7] = k  # store child's parent pointer

        if tree[k,4]:  # check leaf flag
            skip += 1  # add to skip counter
        else:
            c += 1  # count child
            if c + 1 <= sum2n - skip:  # make sure we don't have too many nodes
                data[0,c] = r_X  # store child data
                data[1,c] = r_Y
                tree[k,6] = c  # store child pointer
                tree[c,7] = k  # store child's parent pointer

    return tree

def treeRead(X_Test, tree):
    '''
    Parameters:
    X_Test - The feature data for testing
    tree - the decision tree


    tree cheat sheet:
    tree - | 1 feature | 2 theta | 3 info gain | 4 left flag | 5 right flag |
    6 left child | 7 right child | 8 parent | 9 left prediction | 10 right prediction |

    Output:
    pred - a vector of predictions based on feature data
    '''

    pred = np.zeros((X_Test.shape[0],1))  # initialize prediction vector to # of samples

    for k in np.arange(X_Test.shape[0]):  # iterate over all samples
        c = 0  # initialize tree node counter
        node = 0  # initialize feature
        while c <= tree.shape[0]:  # iterate over nodes
            c += 1  # add to node counter
            if X_Test[k,tree[node,0].astype(int)] < tree[node,1]:  # check if left branch
                node_new = tree[node,5].astype(int)  # set next node
                if node_new == 0 or tree[node,3]:  # check for end of tree
                    pred[k] = tree[node,8]  # make prediction
                    c = tree.shape[0] + 1  # end while loop
                else:
                    node = node_new  # set node for next loop

            if X_Test[k,tree[node,0].astype(int)] >= tree[node,1]:  # check if right branch
                node_new = tree[node,6].astype(int)  # set next node
                if node_new == 0 or tree[node, 4]:  # check for end of tree
                    pred[k] = tree[node, 9]  # make prediction
                    c = tree.shape[0] + 1  # end while loop
                else:
                    node = node_new  # set node for next loop

    return pred





