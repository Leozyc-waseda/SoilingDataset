close all;

[x,y] = meshgrid(0:90, 1:75);

xxx = x;
yyy = x;
zzz = x;

for i = 1:91,
    for j = 1:75,
        if Lx(j,i) > 2.5
            xxx(j,i) = 2.5;
        else
            xxx(j,i) = Lx(j,i);
        end;
        if Ly(j,i) > 4.5
            yyy(j,i) = 4.5;
        else
            yyy(j,i) = Ly(j,i);
        end;
        if Lz(j,i) > 10.0
            zzz(j,i) = 10.0;
        else
            zzz(j,i) = Lz(j,i);
        end;
    end;
end;

figure;
z = -xxx;
minval = min(min(xxx));
surf(x,y,z)
axis([0 90 1 75 -2.5 -minval -2.5 -minval])
hold on;

figure;
z = -yyy;
minval = min(min(yyy));
surf(x,y,z)
axis([0 90 1 75 -4.5 -minval -4.5 -minval])
hold on;

figure;
z = -zzz;
minval = min(min(zzz));
surf(x,y,z)
axis([0 90 1 75 -10.0 -minval -10.0 -minval])
hold on;