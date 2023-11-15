function pfmwrite(fname, img)
%function pfmwrite(fname, img)
%
% Write an image in PFM format to disk.

% $Id: pfmwrite.m 6067 2005-12-20 19:08:27Z rjpeters $
% $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/matlab/pfmwrite.m $

% This matlab code achieves the same effect as the MEX version of this
% file, but is about 1.5x slower:
pfmwritematlab(fname, img);

% Note that matlab is smart about which implementation to choose: if
% there is a file named "pfmwrite.mexglx" (.mexglx for linux) in the
% same directory as this "pfmwrite.m" file, then matlab will use that
% mex function instead of the matlab script version. However, if no
% mex file exists, then the script version will be used.
