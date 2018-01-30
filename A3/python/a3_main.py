import numpy as np
import a3_module as am
import matplotlib.pyplot as plt



# Assignment 3
# Finn Womack

################
# Loading data #
################

Train = np.genfromtxt('knn_train.csv', delimiter=',')  # Loads the Training data
Test = np.genfromtxt('knn_test.csv', delimiter=',')  # loads the Test data

#############################################
# Split independent and dependent variables #
#############################################

Y_Train = Train[:,0]  # loads the training class data into a vector
Y_Test = Test[:,0]  # load the testing class data into a vector

X_Train = Train[:,1:]  # load the training feature data into an array
X_Test = Test[:,1:]  # load the testing feature data into an array

##################
# Normalize data #
##################

for n in np.arange(X_Train.shape[1]):
    Min_Train = np.amin(X_Train[:,n])
    Range_Train = np.amax(X_Train[:,n]) - np.amin(X_Train[:,n])

    X_Train[:,n] = X_Train[:,n] - Min_Train
    X_Train[:,n] = 10 * X_Train[:, n] / Range_Train

    X_Test[:, n] = X_Test[:, n] - Min_Train
    X_Test[:, n] = 10 * X_Test[:, n] / Range_Train

###############
# Problem 1.2 #
###############

K = np.arange(1,70,2)  # set up a vector of K values
Train_Error = np.zeros(K.shape[0])  # initialize a vector for training error
eps = np.zeros(K.shape[0])  # initialize a vector for cross validation error
Test_Error = np.zeros(K.shape[0])  # initialize a vector for cross validation error

for n in np.arange(K.shape[0]):  # loop through all K values
    P_Train = am.knnPred(X_Train, Y_Train, X_Train, K[n])  # predict for training data
    Train_Error[n] = sum((P_Train != Y_Train))/Y_Train.shape[0]  # sum errors and normalize to [0,1]
    for m in np.arange(X_Train.shape[0]):  # loop through each sample to leave out
        ind = np.ones((X_Train.shape[0],), bool)  # make a vector of True values
        ind[m] = False  # set mth value to False
        X2 = X_Train[ind,:]  # Subset X to leave one out
        Y2 = Y_Train[ind]  # Subset Y to leave one out
        P2 = am.knnPred(X2, Y2, np.array([X_Train[m,:]]), K[n])  # predict on the sample left out
        eps[n] = eps[n] + (P2 != Y_Train[m])  # add errors to cross validation error
    eps[n] = eps[n] / (m+1)  # average cross validation error
    P_Test = am.knnPred(X_Train, Y_Train, X_Test, K[n])  # predict for testing data
    Test_Error[n] = sum(P_Test != Y_Test)/Y_Test.shape[0]  # sum errors and normalize to [0,1]

# Note: I normalized the testing and training error so that it would be
# on the same scale as the cross validation error

plt.plot(K, Train_Error, 'bo-', K, eps, 'go-', K, Test_Error, 'mo-') # Plot the 3 error to compare
plt.xlabel('K')
plt.ylabel('Percent missed')
plt.title('Error Ratios on K-NN Predictions')
plt.show()

####################
# Unnormalize Data #
####################

Train = np.genfromtxt('knn_train.csv', delimiter=',')  # Loads the Training data
Test = np.genfromtxt('knn_test.csv', delimiter=',')  # loads the Test data

Y_Train = Train[:,0]  # loads the training class data into a vector
Y_Test = Test[:,0]  # load the testing class data into a vector

X_Train = Train[:,1:]  # load the training feature data into an array
X_Test = Test[:,1:]  # load the testing feature data into an array

###############
# Problem 2.1 #
###############

Stump_1 = am.treePlanter(X_Train, Y_Train, 1)  # create a decition stump
pred_Test = am.treeRead(X_Test, Stump_1);  # make predictions on testing data from the stump
pred_Train = am.treeRead(X_Train, Stump_1);  # make predictions on training data from the stump
test_error = sum(Y_Test != pred_Test)  # sum the total errors made on testing data
train_error = sum(Y_Train != pred_Train)  # sum the total errors made on training data

###############
# Problem 2.2 #
###############

d = np.arange(10)+1  # initialize the depth vector
test_acc = np.zeros((1,len(d))).flatten()  # initialize the testing accuracy vector
train_acc = np.zeros((1,len(d))).flatten()  # initialize the training accuracy vector

for a in np.arange(len(d)):  # iterate over every depth parameter
    tree = am.treePlanter(X_Train, Y_Train, d[a])  # generate tree
    pred = am.treeRead(X_Test, tree)  # make predictions on testing data
    test_acc[a] = sum(sum(np.transpose(Y_Test) != np.transpose(pred)))/len(Y_Test)  # total errors and normalize to [0,1]
    pred = am.treeRead(X_Train, tree)  # make predictions on training data
    train_acc[a] = sum(sum(np.transpose(Y_Train) != np.transpose(pred)))/len(Y_Train)  # total errors and normalize to [0,1]

plt.plot(d, test_acc, 'bo-', d, train_acc, 'mo-')
plt.xlabel('Depth')
plt.ylabel('Error Rate')
plt.title('Error Rate of Predictions vs Tree Depth')
plt.show()

#############
# Problem 3 #
#############

Best_Depth = test_acc.argsort()[0]  # Find the depth with the lowest testing error

######################
# Find Features Used #
######################

tree = am.treePlanter(X_Train, Y_Train, Best_Depth)  # grow tree with best depth
Used_Features = [x for x in set(tree[:,0].astype(int))]  # find the features used in the tree


###############
# Subset data #
###############

X_Train = X_Train[:,Used_Features]
X_Test = X_Test[:,Used_Features]


####################
# Renormalize data #
####################

for n in np.arange(X_Train.shape[1]):
    Min_Train = np.amin(X_Train[:,n])
    Range_Train = np.amax(X_Train[:,n]) - np.amin(X_Train[:,n])

    X_Train[:,n] = X_Train[:,n] - Min_Train
    X_Train[:,n] = 10 * X_Train[:, n] / Range_Train

    X_Test[:, n] = X_Test[:, n] - Min_Train
    X_Test[:, n] = 10 * X_Test[:, n] / Range_Train


#########################################
# Replot knn with reduced feature space #
#########################################

K = np.arange(1,70,2)  # set up a vector of K values
Train_Error = np.zeros(K.shape[0])  # initialize a vector for training error
eps = np.zeros(K.shape[0])  # initialize a vector for cross validation error
Test_Error = np.zeros(K.shape[0])  # initialize a vector for cross validation error

for n in np.arange(K.shape[0]):  # loop through all K values
    P_Train = am.knnPred(X_Train, Y_Train, X_Train, K[n])  # predict for training data
    Train_Error[n] = sum((P_Train != Y_Train))/Y_Train.shape[0]  # sum errors and normalize to [0,1]
    for m in np.arange(X_Train.shape[0]):  # loop through each sample to leave out
        ind = np.ones((X_Train.shape[0],), bool)  # make a vector of True values
        ind[m] = False  # set mth value to False
        X2 = X_Train[ind,:]  # Subset X to leave one out
        Y2 = Y_Train[ind]  # Subset Y to leave one out
        P2 = am.knnPred(X2, Y2, np.array([X_Train[m,:]]), K[n])  # predict on the sample left out
        eps[n] = eps[n] + (P2 != Y_Train[m])  # add errors to cross validation error
    eps[n] = eps[n] / (m+1)  # average cross validation error
    P_Test = am.knnPred(X_Train, Y_Train, X_Test, K[n])  # predict for testing data
    Test_Error[n] = sum(P_Test != Y_Test)/Y_Test.shape[0]  # sum errors and normalize to [0,1]


# Note: I normalized the testing and training error so that it would be
# on the same scale as the cross validation error

plt.plot(K, Train_Error, 'bo-', K, eps, 'go-', K, Test_Error, 'mo-') # Plot the 3 error to compare
plt.xlabel('K')
plt.ylabel('Percent missed')
plt.title('Error Ratios on K-NN Predictions with Reduced Features')
plt.show()

