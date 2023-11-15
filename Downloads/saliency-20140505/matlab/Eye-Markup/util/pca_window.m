function [result,r] = pca_window(result,winp)
%Function PCA_window
%performs the sliding window at multiple time scales
%*****************************************************
if(isfield(result,'len'))
  r = zeros(length(winp), result.len);
  len = result.len;
else
  r = zeros(length(winp), length(result));
end


for j = 1:length(winp)
    st = max(floor(winp(j)/2), 1);
    for i = st:result.len-st
        tmp = result.xy(:,i-st+1:i+st)';
	[Ev,El] = simplepca(tmp);
        if ((max(El)~=0) && (length(El) > 1))
            r(j,i) = min(El)./max(El);
        else
            r(j,i) = 0;
        end
    end
end
%average the windows
r = sum(r)./length(winp);

%Function simplepca
%calculates the eigenvalues and vectors of a 2-d eye trace
%*****************************************************
function [EM,EV] = simplepca(x)
%run a simple pca on the data x and return the components in vector EM and
%the associated variance in EV
%NB: stats toolbox will have princomp and pcares available for this
%functionality; here we simply assume no toolboxes 
x(isnan(x)) = 0;
%x is a cases X sample pnts matrix

[cases, vars] = size(x);

if ~exist('bsxfun')
  x = x - repmat(mean(x),size(x,1),1);
else % easy optimization for at least R2008a and above
  x = bsxfun(@minus,x,mean(x));
end

D = cov(x);%mean removed normalized covarience 
[EM,EV] = eig(D); %EM = eigenvectors EV = eigenvalues 
EM = EM * sqrt(EV); %scaled eigen vectors
EV = diag(EV);
