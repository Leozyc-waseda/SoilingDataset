function val = getvalue(str,vars)
%returns the value of a varargin structure given a string identifier
if(isfield(vars,lower(str)))
  val = vars.(lower(str));
  %val = getfield(vars,lower(str));
else
  error('getvalue:badField', '%s is not a valid field', str)  
end

%ideally should have some sort of confusion resolving or error
%handling code here