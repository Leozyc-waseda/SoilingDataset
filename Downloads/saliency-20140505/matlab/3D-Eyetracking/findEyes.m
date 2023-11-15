function [le,re]=findEyes(pl,pr)

a = 10 ;
b = 15 ;

lx1 = pl(1,1);
lx2 = pl(1,2);
lx3 = pl(1,3);
lx4 = pl(1,4) ;
ly1 = pl(2,1);
ly2 = pl(2,2);
ly3 = pl(2,3);
ly4 = pl(2,4) ;

rx1 = pr(1,1);
rx2 = pr(1,2);
rx3 = pr(1,3);
rx4 = pr(1,4) ;
ry1 = pr(2,1);
ry2 = pr(2,2);
ry3 = pr(2,3);
ry4 = pr(2,4) ;

elx = -((a*(a/(rx4 - rx3) + 1)*(lx1/(lx1 - lx2) + rx1/(rx2 - rx1)) - a*(a/(rx1 - rx2) + 1)*(lx4/(lx4 - lx3) + rx4/(rx3 - rx4)))/((a/(lx2 - lx1) - 1)*(a/(rx4 - rx3) + 1) - (a/(lx3 - lx4)- 1)*(a/(rx1 - rx2) + 1)));
ely = -(((lx1 - lx2)*((a*(a - rx3 + rx4)*(ly1/(lx1 - lx2) - ry1/(rx1 - rx2)))/(rx3 - rx4) + (a*(a + rx1 - rx2)*(-(ly4/(lx3 - lx4)) + ry4/(rx3 - rx4)))/(rx1 - rx2)))/(a*(1 + ((lx1 - lx2)*(a - lx3 + lx4 + rx1 - rx2))/((lx3 - lx4)*(rx1 - rx2)) + (a + lx1 - lx2)/(-rx3 + rx4))));
erx = ((a*lx4)/(lx3 - lx4) + (a*rx4)/(-rx3 + rx4) + ((-1 + a/(lx3 - lx4))*((-a)*(1 + a/(rx1 - rx2))*(lx4/(-lx3 + lx4) + rx4/(rx3 - rx4)) +  a*(lx1/(lx1 - lx2) + rx1/(-rx1 + rx2))*(1 + a/(-rx3 + rx4))))/((-(-1 + a/(lx3 - lx4)))*(1 + a/(rx1 - rx2)) + (-1 + a/(-lx1 + lx2))*(1 + a/(-rx3 + rx4))))/(1 + a/(-rx3 + rx4));
ery=((rx1 - rx2)*(-((a*ly1)/(lx1 - lx2)) + (a*ry1)/(rx1 - rx2) - ((a + lx1 - lx2)*((a*(a - rx3 + rx4)*(ly1/(lx1 - lx2) - ry1/(rx1 - rx2)))/(rx3 - rx4) + (a*(a + rx1 - rx2)*(-(ly4/(lx3 - lx4)) + ry4/(rx3 - rx4)))/(rx1 - rx2)))/(a*(1 + ((lx1 - lx2)*(a - lx3 + lx4 + rx1 - rx2))/((lx3 - lx4)*(rx1 - rx2)) + (a + lx1 - lx2)/(-rx3 + rx4)))))/ (a + rx1 - rx2);

y1 = (ely*(a + lx1 - lx2) - a*ly1)/(lx1 - lx2); 
y4 = ((-ery)*(a - rx3 + rx4) + a*ry4)/(rx3 - rx4);
%tmp = (y4 - y1) /b;
y1-y4
%teta=acos((y1-y4)/b)
%tmp=0;
%sint = sqrt(1-tmp*tmp) ;
teta=0;
elz = (b*(ely - ly1)*(ely - ly4)*sin(teta))/(ly4*y1 - ly1*y4 + ely*(ly1 - ly4 - y1 + y4));
erz = (b*(ery - ry1)*(ery - ry4)*sin(teta))/(ry4*y1 - ry1*y4 + ery*(ry1 - ry4 - y1 + y4));

le=[elx,ely,elz];
re=[erx,ery,erz];