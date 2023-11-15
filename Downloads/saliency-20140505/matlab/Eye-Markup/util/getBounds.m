function [s_beg s_end] = getBounds(foo)
% Utility function: returns the starts and ends of 
% runs of 1s in a logical array as a paired array of indices.
%
% for example:
% [b e] = getBounds([1 0 0 1 1 0 0 1 1 1 0 0 1 1])
% returns b = [1 4 8 13]; e = [1 5 10 14]
% written: John Shen (Oct 2009)
%**************************************************************************

foo = [0 foo 0];
s_beg = find(diff(foo)== 1); 
s_end = find(diff(foo)==-1)-1;
