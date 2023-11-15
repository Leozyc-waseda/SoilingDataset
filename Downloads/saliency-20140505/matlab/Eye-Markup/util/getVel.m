function result = getVel(data,params)
% This utility function recalculates velocity of a trace.
% The velocity is reported in degrees per second
% written: John Shen (Oct 2009)
% iLab - University of Southern California
%**************************************************************************
% data should have xy field which is 2xN
ppd = double(getvalue('ppd',params));
sf = double(getvalue('sf',params));

dR = diff(data.xy,1,2); % diff across coordinates (column by column)
result = [sqrt((dR(1,:)/ppd(1)).^2 + ...
	       (dR(2,:)/ppd(2)).^2) 0] * sf ;
