function out = mylog2(in)
%% This function is to fix the 0*log2(0) issue
  out = log2(in);
  out(~in) = 0;
end