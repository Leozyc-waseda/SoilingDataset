function pgmwrite(fname, img)
%function pgmwrite(fname, img)
%
% Write an image in PGM format to disk.

[fid, msg] = fopen(fname, 'w');
if (fid == -1), error(['Cannot write ' fname ': ' msg]); end
siz = size(img);
fprintf(fid, 'P5\n%d %d\n255\n', siz(2), siz(1));
fwrite(fid, img', 'uint8');
fclose(fid);

