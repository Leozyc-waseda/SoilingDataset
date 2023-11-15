function img = pfmread(fname)
%function img = pfmread(fname)
%
% Read an image in PFM format from disk and return it in a matrix.

% $Id: pfmread.m 6067 2005-12-20 19:08:27Z rjpeters $
% $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/matlab/pfmread.m $

% This matlab code achieves the same effect as the MEX version of this
% file, but is about 2x slower:
img = pfmreadmatlab(fname);

% Note that matlab is smart about which implementation to choose: if
% there is a file named "pfmread.mexglx" (.mexglx for linux) in the
% same directory as this "pfmread.m" file, then matlab will use that
% mex function instead of the matlab script version. However, if no
% mex file exists, then the script version will be used.
