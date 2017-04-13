###############
#  Functions  #
###############

WSolve <- function(x,y){
  a <- solve(t(x) %*% x) %*% (t(x) %*% y)
  return(a)
} 

SseSolve <- function(x,y,w){
  a <- t(y - x %*% w) %*% (y - x %*% w)
  return(a)
}

RandSolve <- function(x,y,n){
  max <- sample(1:(50+n))
  newx <- x
  for(k in 1:n){
    newx <- cbind(newx, runif(dim(x)[1], 0, max[k]))
  }
  w <- WSolve(newx,y)
  sse <- Sse(newx,y,w)
  a <- list(sse,w)
  return(a)
}


###############
#####  1  #####
###############

x <- as.matrix(read.table(file = "housing_train.txt"));
y <- x[,dim(x)[2]];
x <- x[,-dim(x)[2]];
x <- cbind(rep(1, dim(x)[1]), x);

###############
#####  2  #####
###############

w <- WSolve(x,y);

w

###############
#####  3  #####
###############

sse <- SseSolve(x,y,w);
sse

xtest <- as.matrix(read.table(file = "housing_test.txt"));
ytest <- xtest[,dim(xtest)[2]];
xtest <- xtest[,-dim(xtest)[2]];
xtest <- cbind(rep(1, dim(xtest)[1]), xtest);

ssetest <- SseSolve(xtest,ytest,w);
ssetest

###############
#####  4  #####
###############

x2 <- x[,-1];
w2 <- WSolve(x2,y);

xtest2 <- xtest[,-1]

sse2 <- SseSolve(x2,y,w2);
sse2

ssetest2 <- SseSolve(xtest2,ytest,w2);
ssetest2

###############
#####  5  #####
###############

rand <- sample(50:70);

xrand2 <- cbind(x, runif(dim(x)[1], 0, rand[1]), runif(dim(x)[1], 0, rand[2]));
xrand4 <- cbind(x, runif(dim(x)[1], 0, rand[1]), runif(dim(x)[1], 0, rand[2]),
                runif(dim(x)[1], 0, rand[3]), runif(dim(x)[1], 0, rand[4]));
xrand6 <- cbind(x, runif(dim(x)[1], 0, rand[1]), runif(dim(x)[1], 0, rand[2]),
                runif(dim(x)[1], 0, rand[3]), runif(dim(x)[1], 0, rand[4]),
                runif(dim(x)[1], 0, rand[5]), runif(dim(x)[1], 0, rand[6]));
xrand8 <- cbind(x, runif(dim(x)[1], 0, rand[1]), runif(dim(x)[1], 0, rand[2]),
                runif(dim(x)[1], 0, rand[3]), runif(dim(x)[1], 0, rand[4]),
                runif(dim(x)[1], 0, rand[5]), runif(dim(x)[1], 0, rand[6]),
                runif(dim(x)[1], 0, rand[7]), runif(dim(x)[1], 0, rand[8]));
xrand10 <- cbind(x, runif(dim(x)[1], 0, rand[1]), runif(dim(x)[1], 0, rand[2]),
                runif(dim(x)[1], 0, rand[3]), runif(dim(x)[1], 0, rand[4]),
                runif(dim(x)[1], 0, rand[5]), runif(dim(x)[1], 0, rand[6]),
                runif(dim(x)[1], 0, rand[7]), runif(dim(x)[1], 0, rand[8]),
                runif(dim(x)[1], 0, rand[9]), runif(dim(x)[1], 0, rand[10]));

xrandtest2 <- cbind(xtest, runif(dim(xtest)[1], 0, rand[1]), runif(dim(xtest)[1], 0, rand[2]));
xrandtest4 <- cbind(xtest, runif(dim(xtest)[1], 0, rand[1]), runif(dim(xtest)[1], 0, rand[2]),
                runif(dim(xtest)[1], 0, rand[3]), runif(dim(xtest)[1], 0, rand[4]));
xrandtest6 <- cbind(xtest, runif(dim(xtest)[1], 0, rand[1]), runif(dim(xtest)[1], 0, rand[2]),
                runif(dim(xtest)[1], 0, rand[3]), runif(dim(xtest)[1], 0, rand[4]),
                runif(dim(xtest)[1], 0, rand[5]), runif(dim(xtest)[1], 0, rand[6]));
xrandtest8 <- cbind(xtest, runif(dim(xtest)[1], 0, rand[1]), runif(dim(xtest)[1], 0, rand[2]),
                runif(dim(xtest)[1], 0, rand[3]), runif(dim(xtest)[1], 0, rand[4]),
                runif(dim(xtest)[1], 0, rand[5]), runif(dim(xtest)[1], 0, rand[6]),
                runif(dim(xtest)[1], 0, rand[7]), runif(dim(xtest)[1], 0, rand[8]));
xrandtest10 <- cbind(xtest, runif(dim(xtest)[1], 0, rand[1]), runif(dim(xtest)[1], 0, rand[2]),
                 runif(dim(xtest)[1], 0, rand[3]), runif(dim(xtest)[1], 0, rand[4]),
                 runif(dim(xtest)[1], 0, rand[5]), runif(dim(xtest)[1], 0, rand[6]),
                 runif(dim(xtest)[1], 0, rand[7]), runif(dim(xtest)[1], 0, rand[8]),
                 runif(dim(xtest)[1], 0, rand[9]), runif(dim(xtest)[1], 0, rand[10]));

wrand2 <- solve(t(xrand2) %*% xrand2) %*% (t(xrand2) %*% y);
wrand4 <- solve(t(xrand4) %*% xrand4) %*% (t(xrand4) %*% y);
wrand6 <- solve(t(xrand6) %*% xrand6) %*% (t(xrand6) %*% y);
wrand8 <- solve(t(xrand8) %*% xrand8) %*% (t(xrand8) %*% y);
wrand10 <- solve(t(xrand10) %*% xrand10) %*% (t(xrand10) %*% y);

sserand2 <- t(y - xrand2 %*% wrand2) %*% (y - xrand2 %*% wrand2);
sserand4 <- t(y - xrand4 %*% wrand4) %*% (y - xrand4 %*% wrand4);
sserand6 <- t(y - xrand6 %*% wrand6) %*% (y - xrand6 %*% wrand6);
sserand8 <- t(y - xrand8 %*% wrand8) %*% (y - xrand8 %*% wrand8);
sserand10 <- t(y - xrand10 %*% wrand10) %*% (y - xrand10 %*% wrand10);

sserandtest2 <- t(ytest - xrandtest2 %*% wrand2) %*% (ytest - xrandtest2 %*% wrand2);
sserandtest4 <- t(ytest - xrandtest4 %*% wrand4) %*% (ytest - xrandtest4 %*% wrand4);
sserandtest6 <- t(ytest - xrandtest6 %*% wrand6) %*% (ytest - xrandtest6 %*% wrand6);
sserandtest8 <- t(ytest - xrandtest8 %*% wrand8) %*% (ytest - xrandtest8 %*% wrand8);
sserandtest10 <- t(ytest - xrandtest10 %*% wrand10) %*% (ytest - xrandtest10 %*% wrand10);

a <- c(2, 4, 6, 8, 10);
plot(a, c(sserand2, sserand4, sserand6, sserand8, sserand10))
plot(a, c(sserandtest2, sserandtest4, sserandtest6, sserandtest8, sserandtest10))

###############
#####  6  #####
###############

wlam1 <- solve(t(x) %*% x + 0.01*diag(dim(t(x) %*% x)[1])) %*% t(x) %*% y ;
wlam2 <- solve(t(x) %*% x + 0.05*diag(dim(t(x) %*% x)[1])) %*% t(x) %*% y ;
wlam3 <- solve(t(x) %*% x + 0.1*diag(dim(t(x) %*% x)[1])) %*% t(x) %*% y ;
wlam4 <- solve(t(x) %*% x + 0.5*diag(dim(t(x) %*% x)[1])) %*% t(x) %*% y ;
wlam5 <- solve(t(x) %*% x + 1*diag(dim(t(x) %*% x)[1])) %*% t(x) %*% y ;
wlam6 <- solve(t(x) %*% x + 5*diag(dim(t(x) %*% x)[1])) %*% t(x) %*% y ;

sselam1 <- t(y - x %*% wlam1) %*% (y - x %*% wlam1);
sselam2 <- t(y - x %*% wlam2) %*% (y - x %*% wlam2);
sselam3 <- t(y - x %*% wlam3) %*% (y - x %*% wlam3);
sselam4 <- t(y - x %*% wlam4) %*% (y - x %*% wlam4);
sselam5 <- t(y - x %*% wlam5) %*% (y - x %*% wlam5);
sselam6 <- t(y - x %*% wlam6) %*% (y - x %*% wlam6);

sselamtest1 <- t(ytest - xtest %*% wlam1) %*% (ytest - xtest %*% wlam1);
sselamtest2 <- t(ytest - xtest %*% wlam2) %*% (ytest - xtest %*% wlam2);
sselamtest3 <- t(ytest - xtest %*% wlam3) %*% (ytest - xtest %*% wlam3);
sselamtest4 <- t(ytest - xtest %*% wlam4) %*% (ytest - xtest %*% wlam4);
sselamtest5 <- t(ytest - xtest %*% wlam5) %*% (ytest - xtest %*% wlam5);
sselamtest6 <- t(ytest - xtest %*% wlam6) %*% (ytest - xtest %*% wlam6);

b <- c(0.01, 0.05, 0.1, 0.5, 1, 5)
plot(b, c(sselam1, sselam2, sselam3, sselam4, sselam5, sselam6))
plot(b, c(sselamtest1, sselamtest2, sselamtest3, sselamtest4, sselamtest5, sselamtest6))

###############
#####  7  #####
###############

num <- c(0.01, 0.05, 0.1, 0.5, 1, 5)
wmat <- data.frame(w1=wlam1, w2=wlam2, w3=wlam3, w4=wlam4, w5=wlam5, w6=wlam6)


