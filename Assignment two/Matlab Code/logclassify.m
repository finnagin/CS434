function output = logclassify(intput)
% If prediction greater than or equal to 0.5, output = 1;
% otherwise, output = 0

output = (intput >= 0.5);

end