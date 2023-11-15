function I = plotFixationMaps(data,rad,varargin)
%simple program that gets fed three colums by 
%N rows of either saccadic endpoint data, fixation 
%length data or whatever else you want.  For 
%saccadic endpoints use ones in the third column

%data: data
%rad: blurring radius in degrees

varout = parseinputs(varargin);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
scsz = getvalue('screen-size',varout);


I = zeros(scsz(2),scsz(1));
d = [round(data(:,1:2))+1,data(:,3)];
d = d(find(coordsOK(d(:,1),d(:,2))),:);
for (mm = 1:size(d,1))
  I(d(mm,2),d(mm,1)) = I(d(mm,2),d(mm,1)) + d(mm,3);           
end

if (rad ~= 0)
  I = conv2g(I,mean(ppd)*rad);
  %I = reshape(rescale(I(:)',0,255),scsz(2),scsz(1));
end
I = I ./sum(I(:));
gg = imagesc(I);colormap(hot);axis image;
  
function result = mmax(a)
    result = max(max(a));
    




