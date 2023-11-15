function plotMesh(ey,calib)
% function plotMesh(ey)
% plot test grid through full transform (with Spline)

minx = 2500; maxx = 2500; miny = 600; maxy = 600; % for test grid display
divs = 30;

% let's make a test grid and transform it:
minx = min([minx ey(1, :)]); maxx = max([maxx ey(1, :)]);
miny = min([miny ey(2, :)]); maxy = max([maxy ey(2, :)]);

[eytx, eyty] = meshgrid(linspace(minx,maxx,divs+1), ...
			linspace(miny,maxy,divs+1));

eyt = [reshape(eytx, 1, numel(eytx)); ...
       reshape(eyty, 1, numel(eyty))];

% pass the grid through the transforms:
eytc = calibFull(eyt, calib);       % post-TPS affine transform

plot(eytc(1, :), eytc(2, :), '.');
      