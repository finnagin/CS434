function W = gradBatch(X, Y, N, learn, BatSize, Lam)
    W = zeros(1,size(X,2));
    batches = floor(linspace(0,size(X,1),ceil(size(X,1)/BatSize)));
    for c = 1:N
        %cm = mod(c-1,size(batches,2));
        order = randperm(size(X,1));
        for h = 1:(size(batches,2)-1)
            d = zeros(1,size(X,2));
            for k = (batches(h)+1):batches(h+1)
                Y_hat = 1./(1+exp(-W*X(order(k),:)'));
                d = d + (Y(order(k),:)-Y_hat)*X(order(k),:);
            end
            W = W+learn*(d+Lam*sqrt(W*W'));
        end
    end
end