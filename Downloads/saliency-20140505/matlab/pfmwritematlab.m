function pfmwritematlab(fname, img)
%function pfmwritematlab(fname, img)
%
% Write an image in PFM format to disk.

% $Id: pfmwritematlab.m 6066 2005-12-20 19:07:31Z rjpeters $
% $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/matlab/pfmwritematlab.m $

% This matlab code achieves the same effect as the MEX version
% (pfmwrite.mexglx), but is about 1.5x slower:
[fid, msg] = fopen(fname, 'w');
if (fid == -1), error(['Cannot write ' fname ': ' msg]); end
siz = size(img);
fprintf(fid, 'PF\n%d %d\n1.0\n', siz(2), siz(1));
fwrite(fid, img', 'float32');
fclose(fid);
