function img = pgmread(fname)
%function img = pgmread(fname)
%
% Read an image in PGM format from disk and return it in a matrix.

[fid, msg] = fopen(fname, 'r');
if (fid == -1), error(['Cannot read ' fname ': ' msg]); end
siz = fscanf(fid, 'P5\n%d %d\n255\n', 2);
img = fread(fid, [siz(1), siz(2)], 'uint8')';
fclose(fid);

