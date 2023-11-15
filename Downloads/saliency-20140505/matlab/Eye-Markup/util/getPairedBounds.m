function s = getPairedBounds(foo)
% Utility function: returns the starts and ends of 
% runs of 1s in a logical array as a paired array of indices.
%
% for example:
% b = getPairedBounds([1 0 0 1 1 0 0 1 1 1 0 0 1 1])
% returns b = 4x1 struct array
% written: John Shen (Oct 2009)
%**************************************************************************

df = diff([0 foo 0]);
s = struct('beg',num2cell(find(df==1)), ...
    'end',num2cell(find(df==-1)-1));