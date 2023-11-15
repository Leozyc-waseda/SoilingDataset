% $Id: ldepplots.m 6067 2005-12-20 19:08:27Z rjpeters $

% This file generates some figures illustrating the
% link-dependencies in the source tree. To make these plots, do
% three things:
%
% (1) from the shell, run 'make ldepadjacency.m' -- this generates a
%     matlab script named 'ldepadjacency.m' that contains an adjacency
%     matrix as well as a list of filenames
%
% (2) from matlab, run 'ldepadjacency' -- this loads the filenames
%     and adjacency matrix into matlab
%
% (3) from matlab, run this script (matlab/ldepplots.m)

figure(1);
clf;
x=[0.05 0.10 0.75 0.95];
w=0.95*(x(2:end)-x(1:end-1));
y=[0.08 0.33 0.98];
h=0.95*(y(2:end)-y(1:end-1));

subplot('position', [x(2) y(2) w(2) h(2)]);
imagesc(adjacency);
%imagesc(adjacency);
c=hot(356);
c=c(101:356,:);
c(1,:)=0;
colormap(c);
axis tight;
colormap(c);
set(gca, 'XTickLabel', []);
set(gca, 'YTickLabel', []);

subplot('position', [x(1) y(2) w(1) h(2)]);
rng=(0:max(adjacency(:)))';
imagesc(rng);
axis xy;
set(gca, 'YTick', rng+1);
set(gca, 'YTickLabel', rng);
set(gca, 'XTickLabel', []);
hh=ylabel('link dependency level');
set(hh, 'FontSize', 14, 'FontWeight', 'bold');

subplot('position', [x(2) y(1) w(2) h(1)]);
numtargets=sum(adjacency>0);
bar(numtargets,1.0);
axis tight;
axis ij;
box off;
ylabel('# targets depending UPON this one');
hh=xlabel('PREREQUISITE unit');
set(hh, 'FontSize', 18, 'FontWeight', 'bold');
bigones=find(numtargets>250);
for n=1:length(bigones)
    disp(filenames{bigones(n)});
    %hh=text(bigones(n),numtargets(bigones(n)),filenames(bigones(n)));
    %set(hh, 'Rotation', -10-mod(bigones(n),5)*12, 'FontSize', 7, 'Interpreter', 'none');
    %tposx=length(numtargets)+2;
    tposx=min(bigones-20);
    tposy=120+n*22;
    hh=text(tposx,tposy,filenames{bigones(n)});
    set(hh, 'FontSize', 8, 'Interpreter', 'none', ...
            'HorizontalAlignment', 'right');
    hh=line([bigones(n) tposx+2], ...
            [numtargets(bigones(n)) tposy]);
    set(hh, 'Color', [0 0.6 0]);
end

subplot('position', [x(3) y(2) w(3) h(2)]);
barh(sum(adjacency>0,2),1.0);
axis tight;
axis ij;
box off;
xlabel('# prerequisites');
kk=ylabel('TARGET unit');
set(gca, 'YAxisLocation', 'right');
set(kk, 'FontSize', 18, 'FontWeight', 'bold');

set(gcf, 'PaperPositionMode', 'auto');
print(gcf', '-depsc2', 'ldepplot1.eps');

figure(2);
clf;
subplot(1,2,2);
nusers=sum(adjacency>0);
hiusers=find(nusers>50);
barh(nusers(hiusers),1.0);
set(gca, 'YTick', 1:length(hiusers));
set(gca, 'YTickLabel', filenames(hiusers));
axis ij;
axis tight;
xlabel('# targets depending UPON this one');

set(gcf, 'PaperPositionMode', 'auto');
print('-f2', '-depsc2', 'ldepplot2.eps');
