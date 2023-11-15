function newDoit(name, scrw, scrh)
%function doit(name)

if (nargin < 2)
     scrw = 1920;
     scrh = 1080;
end

load(name);
CALIB = newAnalyzeMovieExp(ISCAN, PSYCH, scrw, scrh);
saveCalibTxt(CALIB);

