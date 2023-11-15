function makemovie(df,h,range,sr)
%make a simple movie in a plot window of the eye trace, easy way to check 
%without writing the movie out using ezvision or superimposer

%input: df: path to an eyeS file.  
        %h: 1 for overplot and 0 for not
        %sr: sampling rate used to play video at actual speed if possible.
        %if left blank computer plays as fast as possible.  
        
if (nargin < 2)
    h =0;
end

if (nargin < 3)
    range = 0;
end

if (nargin < 4)
    sr = 0;
else
    sr = 1./sr;
end

corder = {'b.','g.','r.','go','m.','k.','y.'};
% 0 = fixation 
% 4 = smooth pursuit
% 1 = in saccade
% 2 = in blink
% 3 = in blink during a saccade

[dat,pupil] = loadCalibTxt(df);
datax = dat(1,:);
datay = dat(2,:);
datac = dat(3,:);
clear dat;
figure;
set(gcf,'doublebuffer','on');
if (h)
    hold on;
end

if (range == 0)
    range = 1:length(datax);
else 
    range = range(1):range(2);
end

for (i = range)
    plot(datax(i)+1,480- datay(i)+1,corder{datac(i)+1},'markersize',12); 
    axis image;
    set(gca,'xlim',[0 640]);set(gca,'ylim',[0 480]);
    title(num2str(i));
    drawnow;
    pause(sr);
end
