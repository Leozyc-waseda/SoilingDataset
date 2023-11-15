function A = findAffineTransformation(x,y,u,v)
%first component of domain values are all in x, second components are in y
%accordingly, u, and v are first componets and second components of target
%values.

M=zeros(6,6);
M(1,1) = sum(x.*x) ;
M(1,2) = sum(x.*y) ;
M(1,5) = sum(x) ;
M(2,1) = sum(x.*y);
M(2,2) = sum(y.*y);
M(2,5) = sum(y) ;
M(3,3) = sum(x.*x) ;
M(3,4) = sum(x.*y) ;
M(3,6) = sum(x) ;
M(4,3) = sum(x.*y);
M(4,4) = sum(y.*y);
M(4,6) = sum(y);
M(5,1) = sum(x) ;
M(5,2) = sum(y) ;
M(5,5) = length(x) ;
M(6,3) = sum(x) ;
M(6,4) = sum(y) ;
M(6,6) = length(x) ;

N = zeros(6,1) ;
N(1,1) = sum(x.*u) ;
N(2,1) = sum(y.*u) ;
N(3,1) = sum(x.*v) ;
N(4,1) = sum(y.*v) ;
N(5,1) = sum(u) ;
N(6,1) = sum(v) ;
B=M\N;
A=zeros(3,3) ;
A(1,1) = B(1) ;
A(1,2) = B(2) ;
A(1,3) = B(5) ;
A(2,1) = B(3) ;
A(2,2) = B(4) ;
A(2,3) = B(6) ;
A(3,3) = 1 ;
