function f = backupFile(index)
% returns the name of the backup file with index
% if no index returns the global ref with wildcards
% also automatically creates file directory if it does not exist,
% allowing writing

TMP_PFX = 'CALIB';

dir = TMP_PFX(1:find(TMP_PFX=='/',1,'last')-1);
if(isempty(dir))
  dir = tempdir;
elseif(~isdir('dir'))
  mkdir(backupFile('dir'));
end

if(~exist('index'))
  f = sprintf('%s*.mat',TMP_PFX);
elseif(strcmp(index, 'dir'))
  f = [];
elseif(ischar(index))
  f = sprintf('%s%s.mat', TMP_PFX,index);
elseif(index >= 0 & index < 1000)
  f = sprintf('%s%03d.mat',TMP_PFX,index);
else
  error('backupFile:noWrite','Bad index %d out of bounds\n', ...
	index);
end
f = [dir f];





