function M = celltomat(cell)
% NOTE:
% This file is to convert cell array 
% into matrix.
% This file is different with the build in
% function cell2mat.
% Need to improve!

lc = length(cell); % length of the cell array
m = size(cell(end), 1);
M = zeros(m, m);

% obtain number of elements of each vector from the cell array
num = [];
for n = 1:lc
    num = [num, numel(cell{n})];
end

ln = length(num); % obtain the total number of vector in the cell array

% put vector from the cell array into a matrix
for i = 1:ln
    M(1:num(i), i) = vertcat(cell{i}); 
end

end
