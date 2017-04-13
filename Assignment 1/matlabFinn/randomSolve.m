function a = randomSolve(X, Y, Xt, Yt, n, maxs)

if max(size(maxs)) < n
   n =  max(size(maxs));
   warning('Reducing the size of n to match the vactor input...')
end

sample = maxs(1:n);

rmat = repmat(sample,size(X,1),1).*rand(size(X,1),n);
rmat2 = repmat(sample,size(Xt,1),1).*rand(size(Xt,1),n);

X = [X rmat];
Xt = [Xt rmat2];

W = pinv(X'*X)*(X'*Y);

sseTr = (Y-X*W)'*(Y-X*W);
sseTe = (Yt-Xt*W)'*(Yt-Xt*W);

a = [sseTr; sseTe];

end