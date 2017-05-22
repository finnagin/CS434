function image = ReadImage(group, X, k, n)
Img = zeros(n*28, 560);
pos = find(group == k);
ind = randperm(length(pos), n*20);
ite = 0;

while ite < n
    for l = 1:20    
        idx = ind((ite + 1)*l);
        Img((ite*28 + 1):(28*(ite + 1)), (28*l + 1):(28*(l + 1))) = ...
            reshape(X(pos(idx), :), 28, 28);
    end
    ite = ite + 1;
end

image = imshow(Img);

end