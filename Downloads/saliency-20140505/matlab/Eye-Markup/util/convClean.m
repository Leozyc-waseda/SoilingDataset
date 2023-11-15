function res = convClean(vec, fil, shape)
%function res = convClean(vec, fil)
% convolve and return a result that has same size as input
% shape can be 'same', 'padded', 'valid', or 'full'
% NB: valid and full do NOT return results with same size vector
% see help conv2 for more information
% written: John Shen (Oct 2009)
%**************************************************************************


% check that we have enough arguments
error(nargchk(2,3,nargin,'struct'));

if length(vec) < length(fil)
  % do not smooth if vector is too small
  res = vec;
  return;
end

if(nargin < 3) 
  shape = 'same';
end

%filter half-length
fhl = (length(fil) - 1) / 2;
%filter full-length
ffl = length(fil);

if(strcmp(shape,'same') | strcmp(shape,'padded') | strcmp(shape, 'valid'))
%{
if (fhl ~= floor(fhl))
    error('convClean:badArg', 'filter must have odd size'); 
  end
%}
  res = conv(vec, fil);
  if (strcmp(shape,'padded')) % zero-truncated filters 
    for ii = 1:length(fil)
      % fix left tail
      res(ii) = vec(1:ii) * normvec(fil(end:-1:end-ii+1))';
    
      % fix right tail
      res(end-ii+1) = vec(end-ii+1:end) * normvec(fil(1:ii))';
    end
  end
  if(strcmp(shape, 'valid'))
    res = res(ffl:end-ffl+1);
  else %'same'
    % warning: even-size filters may be distorted
    if (fhl == floor(fhl))
      res = res(fhl+(1:length(vec)));
    else
      f = floor(fhl);
      len = length(vec);
      res = 0.5*(res(f:f+len-1) + res(f+1:f+len));
    end
  end
else % 'full' or anything else 
  res = conv(vec,fil);
end

function r = normvec(vec)
%normalize the sum of a vector to 1
r = vec./sum(vec);
