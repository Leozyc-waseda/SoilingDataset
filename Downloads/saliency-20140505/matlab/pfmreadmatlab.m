function img = pfmreadmatlab(fname)
%function img = pfmreadmatlab(fname)
%
% Read an image in PFM format from disk and return it in a matrix.

% $Id: pfmreadmatlab.m 6582 2006-05-11 06:28:22Z rjpeters $
% $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/matlab/pfmreadmatlab.m $

% This matlab code achieves the same effect as the MEX version
% (pfmread.mexglx), but is about 2x slower:

    [fid, msg] = fopen(fname, 'r');
    if (fid == -1)
        error(['Cannot read ' fname ': ' msg]);
    end

    % get the header and the width, height values
    args = fscanf(fid, 'PF\n%d %d\n', 3);
    w=args(1); h=args(2);

    % read (and discard) any optional tags in the file
    while fpeekc(fid) == '!'
        dummytag=fgetl(fid)
        dummyvalue=fgetl(fid)
    end

    % get the max gray value
    spc=fscanf(fid, '1.0%1c', 1);
    if (~isspace(char(spc)))
        error('expected a whitespace character');
    end

    % get the actual image data
    [img, count] = fread(fid, [w h], 'float32');
    if (count ~= w*h)
        error('premature EOF');
    end

    % transpose because pfm data is row-major, but matlab uses
    % column-major format:
    img=img';
    fclose(fid);


function c = fpeekc(fid)
    c = fscanf(fid, '%1c', 1);
    status = fseek(fid, -1, 0);
    if status ~= 0
        error('fseek failed');
    end
