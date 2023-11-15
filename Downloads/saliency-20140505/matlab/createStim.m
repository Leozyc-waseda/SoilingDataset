function createStim(filename,period,scr_size,varargin)
%creates a a file representing a 2d stimulus from 1d (time varying)
%signals
%
%input:
%filename: text file for output
%period:sampling of stimulus in seconds
%scr_size: screen size of 2d stimulus eg. [xx yy]
%varargin: a list of 1d signals and screen locations 
%eg. signal,xpos,ypos
%example: x and x1 are two 1d signals of the same length sampled at 1000Hz
%createStim('test.txt',.001,[640 480],x,320,240,x1,350,240);

X = [];
for (ii = 1:3:length(varargin))
  x = varargin{ii};
  px = varargin{ii+1};
  py = varargin{ii+2};
  sz = size(x);
  if (sz(1) == 1)
    x = x';
  end
  X = [X,x,repmat(px,[length(x),1]),repmat(py,[length(x),1])];
end

ferp = fopen(filename,'w');
fprintf(ferp,[num2str(period),'\n']);
fprintf(ferp,[num2str(scr_size(1)),'x',num2str(scr_size(2)),'\n']);
fclose(ferp);

dlmwrite(filename, X, 'delimiter', ' ', '-append');


  