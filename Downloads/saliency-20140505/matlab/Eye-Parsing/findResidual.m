function r = findResidual(x,y,u,v,A)

n = length(x) ;
r = 0 ;
for i=1:n
    a=[x(i);y(i);1] ;
    b=[u(i);v(i);1] ;
    c = A*a ;
    r = r + sum((b-c).*(b-c)) ;
    
end