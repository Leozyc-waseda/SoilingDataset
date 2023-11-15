function copytonewfig
%copies the current axes to a new figure and resizes it to take up the
%entire figure
%
%returns the handle to the new figure. 
g = gca;
f = figure;
copyobj(g,f);
g = findobj('type','axes','parent',f);
set(g,'position',[0.13 0.11 0.775 0.815]);
set(g, 'fontsize',14);