function [le,re] = doAllInOne(Psych,Eye)

ppcx = 21.51 ;%this is pixel per centimeter along x axis (vertical)
ppcy = 21.6 ; %this is pixel per centimeter along y axis (horizontal)
k=44.6 ; 

[pl,pr] = inspectSessions(Psych,Eye);
pl(1,:) = pl(1,:)/ppcx;
pl(2,:) = pl(2,:)/ppcy;
pr(1,:) = pr(1,:)/ppcx;
pr(2,:) = pr(2,:)/ppcy ;
[al,ar] = adjust(pl,pr);


[le , re] = locateEyes(al,ar,k);