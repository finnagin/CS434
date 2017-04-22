function W = gradBatch(X, Y, N, learn, Lam, loss)
%% Parameters:
    % X - An array containing the feature vectors from each of the samples
    % Y - A vector containing the class to be predicted for each sample
    % N - The number of iterations
    % learn - the learning rate
    % Lam - the lambda used for regularization
    %       (if no regularization is desired use Lam = 0)
    % loss - A optional indicator for the type of output desired
                % If loss != 1 or is not inputed then the end W vector is
                % outputted
                % If loss = 1 then instead of the W being outputted a
                % vector of the loss function for every iteration is
                % outputted
                
%% code:


    if (~exist('loss','var')) % This sets the loss indicator to zero if nothing 
                              % is inputted for loss
        loss = 0; 
    elseif (loss ~= 1 && loss ~= 0) % this sets the loss to zero if a number
                                    % besides 0 or 1 is inputted for loss
        loss = 0;
        disp('Warning: setting loss flag to 0.') % displays a warning to 
                                                 % indicate that the loss 
                                                 % indicator has been changed
    end
    if loss
       losses = zeros(1,N); % initializes the loss vector
    end
    W = zeros(1,size(X,2)); % initializes W
    for c = 1:N
        d = zeros(1,size(X,2)); % resets d to the zero vector
        for k = 1:size(X,1) % loops through allsmples
            % updates d for each sample
            Y_hat = 1./(1+exp(-W*X(k,:)'));
            d = d + (Y(k,:)-Y_hat)*X(k,:);
        end
        W = W+learn*(d+Lam*W); % updates W with regularization at the 
                               % desired learning rate
        if loss
            losses(c) = lossFun(X, Y, W); % puts loss value into the vector
        end
        if norm(d)<1e-16 && c>100 && loss % checks if d is really small
                                          % i.e. W is not changing much for
                                          % each iteration
            losses((c+1):N) = repmat(losses(c),1,(N-c)); % repeats the latest entry
                                                         % so that the loss
                                                         % vector is the
                                                         % correct size
            W = losses; % sets output to the loss vector
            return % returns the output
        elseif norm(d)<1e-16 && c>100 % checks if d is really small
            return % return current W
        end
    end
    if loss
       W = losses; % outputs loss vector
    end
end