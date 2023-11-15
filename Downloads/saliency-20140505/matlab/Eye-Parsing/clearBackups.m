function clearBackups(index)
% This function will delete all the backups and the folder they are
% contained in.
% If index is specified it will only delete the specific one/ones

if(~exist('index'))
  delete(backupFile());
  % if directory is empty, clear it  
  if(isempty(ls(backupFile('dir'))))
    rmdir(backupFile('dir'));
  end
elseif isnumeric(index)
  for i = 1:numel(index)
    % TODO: should check to see if file exists
    delete(backupFile(index(i)));
  end
end