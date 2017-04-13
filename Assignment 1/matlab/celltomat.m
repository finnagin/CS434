function M = celltomat(k, cell)
% NOTE:
% This file is to convert cell array 
% into matrix.
% This file is different with the build in
% function cell2mat.
% Need to improve!

M = []; % construct an empty matrix

for i = 1:k
    M = [M cell{i}];
end

end
