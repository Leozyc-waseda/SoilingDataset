function sacPlots(glob,E,varargin)
%gives a bunch of plots about various saccade statistics
%
%glob: a directory to the files with an extenion ie. /lab/dberg/*.eye
%E: is an array or scalar of events to analyze
%varargin: a list of arguments SEE parseinputs.m
%
% For now this is really just build for saccades but later will be for
% other event types

varout = parseinputs(varargin);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
sacfilt = getvalue('sac-filter',varout);
scsz = getvalue('screen-size',varout);

h = waitbar(0,'Loading...');

[filez,glob] = strip_file_path(glob);
data = [];
for (ii = 1:length(filez))
    fnam = [glob,filez{ii}];
    data{ii} = loadCalibTxt(fnam); %load the file
    x = ii./length(length(filez));
    h = waitbar(x,h);
end
close(h);


h = waitbar(0,'Calculating...');
for (ii = 1:length(E))
    etime = [];
    result = [];
    etimec = [];
    resultc = [];
    isitime = [];
    for (jj = 1:length(data))
        h = waitbar(jj./length(data),h);
        temp = [];
        sz = size(data{jj});
        %to be used for velocity peak detection
        ff = sacfilt/sf;%create the filter for 20Hz (Normalized freq)
        [b a] = butter(4,ff,'low');
        temp(1,:) = ( filtfilt(b,a,data{jj}(1,:)) ./ ppd(1)) .* sf;
        temp(2,:) = ( filtfilt(b,a,data{jj}(2,:)) ./ ppd(2)) .* sf;
        %compute velocity in deg/Second
        %take the difference of successive values
        result2 = [temp(1:2, 2:sz(2)) [0;0]]; diff = temp(1:2,:) - result2;
        % compute velocity in deg per sec:
        %        vel = sqrt(diff(1, :).^2 + diff(2, :).^2)) ./ ppd) .* sf;
        vel = (sqrt(diff(1, :).^2 + diff(2, :).^2));
        % cleanup the start/end:
        vel(1:length(ff)) = 0; vel(length(vel)-length(ff)+1:length(vel)) = 0;

        %lets also get the intersaccadic intervals in ms
        [isib,isie] = findEvent(data{jj},[0 2 4 5]);
        time = (isie-isib);
        isitime = [isitime,time .* (1000/sf)];

        %find event
        %[sb,se] = findEvent(data{jj},E(ii));
        [sb,se] = findEvent(data{jj},1,1);%find saccades not including
        %combined
        [sba,sea] = findEvent(data{jj},1);%find saccades
        eventname = getlabel(E(ii));

        if (isSaccade(E(ii)))
            %compute the saccade target positions
            %we need some code here two function calls one with and one
            %without combined
            [et,r] = getdata(data{jj},sb,se,vel,varargin);
            etimec = [etime,et];
            resultc = [result,r];
            [et,r] = getdata(data{jj},sba,sea,vel,varargin);
            etime = [etime,et];
            result = [result,r];
        end%end if saccade
    end%end data

    mainm = resultc;%save mainm.mat mainm
    %break;break;break;
    
    close(h);
    %Now that we have read in all the data for an event lets make some
    %plots
  %  figure;
  %  subplot(3,2,1);
  %  xscale = 0:3:max(etimec);%3 ms bins
  %  [durh,durx] = histc(etimec,xscale);
  %  durh = durh./sum(durh);
  %  bar(xscale+1.5,durh);hold on;
  %  xlabel('Duration (ms)');
  %  ylabel('Probability');
  %  title(eventname);
  %  handles.fig{ii}(1) = gca;
  %  set(gca, 'ButtonDownFcn', 'copytonewfig');

    %saccade specific plots
    if (isSaccade(E(ii)))
        %plot the endpoints as a heat map
%        subplot(3,2,2);
%        I = zeros(scsz(2),scsz(1));
%        ci = round(result(1,:))+1;%x pos
%        ri = round(result(2,:))+1;%y pos
%        f = find((ci < 1) | (ci > scsz(1)));%x pos
%        ci(f) = [];
%        ri(f) = [];
%        f = find((ri < 1) | (ri > scsz(2)));%x pos
%        ci(f) = [];
%        ri(f) = [];
%        for(ll = 1:length(ri))
%            I(ri(ll),ci(ll)) = I(ri(ll),ci(ll))+1;
%        end
%        rad = 2.5;
%        I = conv2g(I,ppd*rad);
%        I = reshape(rescale(I(:)',0,255),scsz(2),scsz(1));
 %
 %   gg = imagesc(I); colormap(hot); %axis image;
%	set(gca, 'XTick', []);
%	set(gca, 'YTick', []);
%        set(gca, 'ButtonDownFcn', 'copytonewfig');
%        title('Saccadic Endpoint Distribution');

                 %plot(ci,ri,'bo');
                 %title('Saccadic Endpoint Distribution');
                 %handles.fig{ii}(2) = gca;
                 %set(gca,'xlim',[0 scsz(1)]);
                 %set(gca,'ylim',[0 scsz(2)]);
                 %set(gca, 'ButtonDownFcn', 'copytonewfig');

        %histogram of the amplitudes
 %       subplot(3,2,2)
 %       xscale = 0:.5:((1./ppd)*scsz(1));%.5 degree intervals at screen extent
 %       [durh,durx] = histc(result(3,:),xscale);
 %       durh = durh./sum(durh);
 %       bar(xscale+.25,durh);hold on;
 %       xlabel('Amplitude(Deg)');
 %       ylabel('Probability');
 %       title('Saccadic Amplitude Histrogram');
 %       handles.fig{ii}(3) = gca;
 %      set(gca, 'ButtonDownFcn', 'copytonewfig');

        %plot the velocity histogram
 %       subplot(3,2,3);
 %       xscale = 0:5:max(resultc(4,:));
 %       [durh,durx] = histc(resultc(4,:),xscale);
 %       durh = durh./sum(durh);
 %       bar(xscale+2.5,durh);hold on;
 %       xlabel('Peak Velocity (Deg/Sec)');
 %       ylabel('Probability');
 %       title('Saccadic Velocity Histrogram');
 %       handles.fig{ii}(4) = gca;
 %       set(gca, 'ButtonDownFcn', 'copytonewfig');

        %plot the main sequence
        subplot(3,2,4);
        plot(resultc(3,:),resultc(4,:),'ro');
	hold on;
	x = [0, max(resultc(3,:))];
	b = regress(resultc(4,:)', [resultc(3,:)' ones(length(resultc(3,:)),1)]);
	y = [x' ones(2,1)]*b;
	h = plot(x, y, 'b');
	set(h, 'LineWidth',2)
	hold off;
        title(['Main Sequence: N = ',num2str(length(resultc(3,:)))...
		', slope =', num2str(b(1),3)]);
        xlabel('Amplitude(Deg)');
        ylabel('Peak Velocity (Deg/Sec)');
        handles.fig{ii}(5) = gca;
        set(gca, 'ButtonDownFcn', 'copytonewfig');

        %plot the intersaccdic interval
        subplot(3,2,5);
        xscale = 0:3:max(isitime);%3 ms bins
        [durh,durx] = histc(isitime,xscale);
        durh = durh./sum(durh);
        bar(xscale+1.5,durh);hold on;
        xlabel('Duration (ms)');
        ylabel('Probability');
        title('Intersaccadic Interval');
        handles.fig{ii}(6) = gca;
        set(gca, 'ButtonDownFcn', 'copytonewfig');
    end%end saccade specific plots

end% end events

function [etime,result] = getdata(data,sb,se,vel,varargin)
varout = parseinputs(varargin);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);

result = zeros(4,length(sb));
etime = [];
for kk = 1:length(sb)
    [eb,ee,edur] =computeEndPoints(data,sb(kk),se(kk),...
        varargin);
    %duration
    etime = [etime,edur .* (1000/sf)];
    % targetx pixels
    result(1,kk) = ee(1);
    % targety pixels
    result(2,kk) = ee(2);
    % calculate amplitude in degrees
    result(3,kk) = ...
        sqrt( ((ee(1) - eb(1))./ppd(1)).^2 + ((ee(2) - eb(2))./ppd(2)).^2 );
    %find the peak valuein the interval
    result(4,kk) = max(vel(sb(kk):se(kk)));
end %end event instance


function result = mmax(a)
    result = max(max(a));
