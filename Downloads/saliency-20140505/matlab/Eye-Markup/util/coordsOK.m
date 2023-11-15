function r = coordsOK(x,y,varargin)
%returns 1's in places where cords are ok. 

varout = parseinputs(varargin);
scsz = getvalue('screen-size',varout);

f = find( (x > scsz(1)) | (y > scsz(2)) | (x < 1) | (y < 1) );
r = ones(size(x));
r(f) = 0;
