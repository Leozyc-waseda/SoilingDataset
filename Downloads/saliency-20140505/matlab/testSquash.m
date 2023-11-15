
% test the squash() function in Image_MathOps.C

a = 0;   % old min
b = 100; % old midpoint
c = 200; % old max
d = 0;   % new min
e = 180; % new midpoint
f = 255; % new max

x = a:(c-a)/100:c;  % input data

q0 = c*c*(b*(b-c)*(b-c)*(-4.0*a*a + 3.0*a*b + 2.0*a*c - b*c)*d + ...
	  a*a*(a-c)*(a-c)*(a-c)*e) + a*a*(a-b)*(a-b)*b* ...
(a*(b-2.0*c) + c*(-3.0*b + 4.0*c))*f;
q1 = 2.0*a*c*(c*c*c*c*(e-d) - 3.0*b*b*b*b*(d-f) + ...
	       4.0*b*b*b*c*(d-f) + 2.0*a*a*a*c*(e-f) + a*a*a*a*(f-e)+ ...
	       2.0*a*(c*c*c*(d-e) + 2.0*b*b*b*(d-f) + ...
		       3.0*b*b*c*(f-d)));
q2 = a*a*a*a*a*(e-f) + a*a*a*a*c*(e-f) + 8.0*a*a*a*c*c*(f-e) + ...
(a+c)*(c*c*c*c*(d-e) + 3.0*b*b*b*b*(d-f) + 4.0*b*b*b*c*(f-d)) - ...
4.0*a*a*(2.0*c*c*c*(d-e) + b*b*b*(d-f) + 3.0*b*c*c*(f-d));
q3 = 2.0*(c*c*c*c*(e-d) + 2.0*a*a*(b-c)*(b-c)*(d-f) + ...
	   2.0*b*b*c*c*(d-f) + 2.0*a*a*a*c*(e-f) + ...
	   b*b*b*b*(f-d) + a*a*a*a*(f-e) + ...
	   2.0*a*c*(c*c*(d-e) + b*b*(d-f) + 2.0*b*c*(f-d)));
q4 = -3.0*a*b*b*d + 2.0*b*b*b*d + 6.0*a*b*c*d - 3.0*b*b*c*d - ...
3.0*a*c*c*d + c*c*c*d + a*a*a*e - 3.0*a*a*c*e + 3.0*a*c*c*e - ...
c*c*c*e - (a-b)*(a-b)*(a + 2.0*b - 3.0*c)*f;

denom = (a-b)*(a-b)*(a-c)*(a-c)*(a-c)*(b-c)*(b-c);
q0 = q0 / denom; q1 = q1 / denom; q2 = q2 / denom;
q3 = q3 / denom; q4 = q4 / denom;

% compute squashing of x:
y = q0 + x.*(q1 + x.*(q2 + x.*(q3 + x.*q4)));

plot(x,y,'-'); hold on;
axis([a-(c-a)*0.2 c+(c-a)*0.2 d-(f-d)*0.2 f+(f-d)*0.2]);
plot([a-(c-a)*0.2 c+(c-a)*0.2], [d d]);
plot([a-(c-a)*0.2 c+(c-a)*0.2], [f f]);
plot([a a], [d-(f-d)*0.2 f+(f-d)*0.2]);
plot([c c], [d-(f-d)*0.2 f+(f-d)*0.2]);
plot([a b c], [d e f], 'r*');
hold off;
