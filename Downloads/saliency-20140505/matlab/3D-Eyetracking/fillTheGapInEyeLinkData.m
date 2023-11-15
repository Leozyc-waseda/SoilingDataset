function out = fillTheGapInEyeLinkData(d,f)

out=[];
ts = 1000/f;

findex=d(3,1);
out=[out, d(:,1)];
for ii=2:length(d);
    sindex = d(3,ii);
    p = (sindex - findex)/ts -1 ;
    if p > 0
        for jj = 1:p
            out=[out,[NaN ;NaN ; NaN]];
        end
    end
    out=[out,d(:,ii)];
    findex=sindex;
end


