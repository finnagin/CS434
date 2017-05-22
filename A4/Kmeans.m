function [group, sse] = Kmeans(X, K)
samples = size(X, 1);
iter = 0;
center = X(randperm(samples, K), :);
idx = zeros(samples, 1);
sse = [0 1];

while abs(sse(end) - sse(end-1)) > 1e-6 && iter < 100
    error = 0;
    for s = 1:samples
        distance = zeros(K, 1);
        for c = 1:K
            distance(c, 1) = norm(X(s,:) - center(c,:));
        end
        [minDist, minIdx] = min(distance);
        idx(s) = minIdx;
    end
    for k = 1:K
        pos = idx == k;
        if size(pos) ~= 0
            center(k,:) = mean(X(pos,:));
            error = error + sum(sum((X(pos,:) - center(k,:)).^2));
        end
    end
    sse = [sse, error];
    iter = iter + 1;
end

sse = sse(3:end);
group = idx;
end