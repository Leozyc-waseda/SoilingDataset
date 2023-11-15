function transl_coords(x,y)
%function transl_coords(x,y)

cx = x*16;
cy = y*16;  % center coords
topx = cx - 32;
topy = cy - 32;
botx = cx + 32;
boty = cy + 32;

disp([ 'top=(' int2str(topx) ', ' int2str(topy) ')  bottom=(' int2str(botx) ...
     ', ' int2str(boty) ')' ]);
     
