function [ey,sc] = findBestSubsetForAffineTrans(X,Y,U,V,n,m);
%T = findAffineTransformation(X,Y,U,V) ;

s1 = zeros(n,1) ;
l=length(X) ;

p=randperm(l) ;
minPerm = randperm(l) ;

x=zeros(n,1);
y=x;u=x;v=x;
minRes = 9999999999 ;
for i=1:m 
    p=randperm(l);
    for j=1:n
        x(j,1) = X(p(j)) ;
        y(j,1) = Y(p(j)) ;
        u(j,1) = U(p(j)) ;
        v(j,1) = V(p(j)) ;
    end
    A = findAffineTransformation(x,y,u,v) ;
    r = findResidual(x,y,u,v,A) ;
    minRes = min(minRes,r) ;
    if minRes == r
        minPerm = p ;
        T=A ;
    end
    
end

ey=zeros(2,n);
sc=zeros(2,n);

for k=1:n
    s1(k,1) = minPerm(k) ;
    ey(1,k) = X(minPerm(k)) ;
    ey(2,k) = Y(minPerm(k)) ;
    sc(1,k) = U(minPerm(k)) ;
    sc(2,k) = V(minPerm(k)) ;
end

%s = sort(s1) ;


