function plotCalib(sc, ey)
%function plotCalib(sc, ey)

% build some vectors going from screen to eye:
sz = size(ey);
p = NaN(sz(1), 3 * sz(2));

p(:, 1:3:3*sz(2)) = sc;
p(:, 2:3:3*sz(2)) = ey;

plot(p(1, :), p(2, :)); 
hold on; 
plot(sc(1, :), sc(2, :), '*');
hold off;
