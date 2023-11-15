function [result,r] = pca_clean(data,ff,winp,params)
%this function uses a simple pca filter to clean the data 
%
%inputs: data: read from an eye or eyeS file
%
%        see help parsinputs for perameter options
%
%output: a newly marked result 3 x samples matrix.  
%
%written: Sept 2006
%author : David Berg
%Ilab - University of Southern California
%**************************************************************************

%varout = parseinputs(var);
sf = getvalue('Sf',params);
newdata = filterTrace(data,params,ff);

%run a simple pca and save the ratio of the principle axis for each window
%size 
%should also make sure there is some variance, otherwise we probably have the
%same points, 

%do an average of a couple of differnt sliding pca windows
[result,r] = pca_window(newdata,winp);
