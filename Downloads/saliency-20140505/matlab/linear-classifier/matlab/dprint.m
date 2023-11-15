function dprint(msg,arg)

[ST,I] = dbstack;

if nargin < 2
    % Create something like disp('<a href = "matlab:open(''linear_model.m'')">hello</a>')
    link = ['<a href = "matlab:open(''' ST(2,1).file ''')">']; 
    disp([link ST(2,1).file '</a> L' num2str(ST(2,1).line) ' ' msg ]);
elseif strcmp(arg,'nolink')
    fprintf('<%s at %d >>>\t%s\n',ST(2,1).file,ST(2,1).line,msg); 
else
    error('Invalid debugging option given');
end