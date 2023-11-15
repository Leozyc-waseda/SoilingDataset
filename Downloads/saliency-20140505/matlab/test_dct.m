 x=-7.5:1:7.5 ;
l=4           

y=sin(x*2*pi/l);z=dct(y);plot(z);hold on;plot(z,'o');grid on;hold off
