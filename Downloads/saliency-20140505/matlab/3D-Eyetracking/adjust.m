function [pl,pr]= adjust(l,r)

pl = l;
pr = r;

pl(2,1) = (l(2,1)+l(2,2))/2;
pl(2,2) = pl(2,1) ;

pr(2,1) = (r(2,1)+r(2,2))/2;
pr(2,2) = pr(2,1) ;

pl(2,3) = (l(2,3)+l(2,4))/2;
pl(2,4) = pl(2,3) ;

pr(2,3) = (r(2,3)+r(2,4))/2;
pr(2,4) = pr(2,3) ;

pl(1,1) = (l(1,1)+r(1,1))/2;
pr(1,1) = pl(1,1);


pl(1,2) = (l(1,2)+r(1,2))/2;
pr(1,2) = pl(1,2);

pl(1,3) = (l(1,3)+r(1,3))/2;
pr(1,3) = pl(1,3);

pl(1,4) = (l(1,4)+r(1,4))/2;
pr(1,4) = pl(1,4);

