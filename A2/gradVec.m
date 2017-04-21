function W = gradVec(X, Y, N, learn, Lam, loss)
    if (~exist('loss','var'))
        loss = 0; 
    elseif (loss ~= 1 && loss ~= 0)
        loss = 0;
        disp('Error: setting loss flag to 0.')
    end
    wvec = zeros(N,size(X,2));
    if loss
       losses = zeros(1,N);
    end
    W = zeros(1,size(X,2));
    for c = 1:N
        d = zeros(1,size(X,2));
        for k = 1:size(X,1)
            Y_hat = 1./(1+exp(-W*X(k,:)'));
            d = d + (Y(k,:)-Y_hat)*X(k,:);
        end
        wvec(c,:) = W;
        W = W+learn*(d+Lam*W);
        if loss
            losses(c) = lossFun(X, Y, W);
        end
        if norm(d)<1e-16 && c>100
            losses((c+1):N) = repmat(losses(c),1,(N-c));
            W = losses;
            return
        end
    end
    W = wvec;
    if loss
       W = losses; 
    end
end