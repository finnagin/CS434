function image = ReadImage(group, X, k, n)
%% Parameters:
    % X - An array containing the data
    % K - The cluster to display
    % group - a vector containing the clusters of X (where the indicies 
    %         correspond to the indicies of the examples in X)
    % n - The number of rows of 20 images to display

%% code:
Img = zeros(n*28, 560); % initialize image
pos = find(group == k); % subset the data by the given cluster
ind = randperm(length(pos), n*20); % take a random sample of the images
ite = 0; %set counter

while ite < n
    for l = 1:20 % iterate through sample
        idx = ind((ite + 1)*l); % set index to correct sample
        Img((ite*28 + 1):(28*(ite + 1)), (28*l + 1):(28*(l + 1))) = ...
            reshape(X(pos(idx), :), 28, 28); % get image ready for display
    end
    ite = ite + 1; % count
end

image = imshow(Img); % display image

end