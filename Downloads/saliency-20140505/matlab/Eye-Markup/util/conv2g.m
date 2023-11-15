function img = conv2g(A,K)
%this implements a K+1 point averaging filter from the binomial distribution
%and convolves the filter with A mainting a truncated boundary condition.
%INPUTS
%A: your 2d matrix
%K the number of points in the binomial kernel


w = size(A,2);
h = size(A,1);
%lets first create our kernel
K = diag(gaussfunc([351 351],[(K) (K)],[0 0]))';
Kn = K./sum(K);
ks = ceil(length(K)./2);

%now the x direction
for (ii = 1:h)
    img(ii,:) = convn(A(ii,:),Kn,'same');    
end    
for (jj = 1:ks-1)
    myfilt = K(ks-jj+1:end);
    myfilt = myfilt ./sum(myfilt);
    img(:,jj) = sum(A(:,1:length(myfilt)) .* repmat(myfilt,h,1),2);   
end
cc = 1;
for (jj = (w-ks+2:w))
    myfilt = K(1:end-cc);
    myfilt = myfilt./sum(myfilt);
    img(:,jj) = sum(A(:,end-length(myfilt)+1:end) .* repmat(myfilt,h,1),2);   
    cc = cc+1;
end

%now the y direction
imgx = img;
for (ii = 1:w)
    img(:,ii) = convn(img(:,ii),Kn','same');    
end    
for (jj = 1:ks-1)
    myfilt = K(ks-jj+1:end);
    myfilt = myfilt./sum(myfilt);
    img(jj,:) = sum(imgx(1:length(myfilt),:) .* repmat(myfilt',1,w));
end
cc = 1;
for (jj = h-ks+2:h)
    myfilt = K(1:end-cc);
    myfilt = myfilt./sum(myfilt);
    img(jj,:) = sum(imgx(end-length(myfilt)+1:end,:) .* repmat(myfilt',1,w));   
    cc = cc+1;
end

function h = gaussfunc(size,sig,pos)     
%a 2d gauss in rows by column, alla matlab style
    [x,y] = meshgrid(1:size(2),1:size(1));
%    x = mod(x,-size(1)+pos(1)-1);
%    y = mod(y,-size(2)+pos(2)-1);
%    x = mod(x,size(2)-pos(2));
%    y = mod(y,size(1)-pos(1)); 
    c = ceil(size./2);
    pos = pos + c;
    arg   = -(((x-pos(2)).^2)./(2*sig(2).^2) + ((y-pos(1)).^2)./(2*sig(1).^2));
    h     = exp(arg);
    h(h<eps*max(h(:))) = 0;    
    h = h./sum(h(:));
    