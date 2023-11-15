function result = findOffScreen(data,params)
%Here we find other nasty eye areas such as loss of tracking
%inputs: data: as described above
%        params : structure created from parseinputs.m
% revised: Nov 2009
% editor : John Shen
ss = getvalue('screen_size',params);
ppd = getvalue('ppd',params);
offnan = getvalue('offscreenmargin', params);
scode = getvalue('code', params);
vblevel = getvalue('verbose',params);
V = getvalue('verboselevels',params);

%now lets find any out of bounds eye movements areas
if offnan
    deg = 1;
else
    deg = 0;
end

result = data;
is_offscreen = (data.xy(1,:) > ss(1)-(deg*ppd(1))) | ...
    (data.xy(1,:) < 0+(deg*ppd(1))) | ...
    (data.xy(2,:) > ss(2)-(deg*ppd(2))) | ...
    (data.xy(2,:) < 0+(deg*ppd(1)));

result.xy(:,is_offscreen) = NaN;
result.pd(:,is_offscreen) = NaN;
result.status(:,is_offscreen) = scode.NC;
result.vel(:,is_offscreen) = NaN;

if vblevel>=V.SUB && any(is_offscreen)
    foo = getBounds(is_offscreen);
    fprintf('\t%d samples in %d tracks cleaned from trace\n', sum(is_offscreen), length(foo));
end