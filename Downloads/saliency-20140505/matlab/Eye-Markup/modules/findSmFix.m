function result = findSmFix(data,params)
%lets run a sliding PCA window over the data and calculate the ratio
%of the largest/smallest principle axis to decide whether we are in 
%fixation or smooth pursuit
%
%inputs:    result : the data premarked by other files                    
%
%see help parseinputs for more options
%written: Sept 2006
%author : David Berg  
%modified: John Shen (Oct 2009)
%iLab - University of Southern California
%*****************************************************************

V = getvalue('verboselevels',params);
vblevel = getvalue('verbose',params);

sf = getvalue('sf',params);%sampling freq
winp = getvalue('smf_window',params);%window sizes in ms
winp = (winp ./ (1000/sf));
ff = getvalue('smf_prefilter',params);%low pass cutoff for filtering
smthresh = getvalue('smf_thresh',params);%threshold for fixation 
scode = getvalue('code',params);

result = data;
[data,r] = pca_clean(data,ff,winp,params);

%find who is below threshold and call that a smooth pursuit
is_sp = (r < smthresh) & (result.status == scode.FIXATION);

if vblevel>=V.SUB
    [sb, se] = getBounds(is_sp);
    fprintf('\t%d smooth pursuits found with %d samples \n', length(sb), sum(is_sp));
    
end
%NB: output does NOT get filtered
result.status(is_sp) = scode.SMOOTH;



