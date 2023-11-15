function c = strjoin(a, sep)
% function c = strjoin(a, sep)

% joins strings together with separators
% assumes a is a 1D cell array of strings
% default for separator is space
if nargin == 1
  sep = ' ';
end
c = '';
for i = 1:numel(a)-1
  c = [c a{i} sep];
end
c = [c a{end}];
