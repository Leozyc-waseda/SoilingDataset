function [pl,pr]=inspectSessions(psych,eyelink)
% this function takes in psych and processed eyelink files and spits out
% the average of projections for both eyes for each of LEDs


theSet = {'LEDCALIB','LEDVALID'};

pl=zeros(2,4);
pr=zeros(2,4);

figure;
hold on ;
for ii=1:length(psych.fname)
    if strfind( psych.fname{ii},'LEDCALIB')==1
        plot(eyelink.data{ii}(2,:),eyelink.data{ii}(3,:));
        plot(eyelink.data{ii}(4,:),eyelink.data{ii}(5,:),'g');
    end
end

figure;
hold on ;
for ii=1:length(psych.fname)
    if strfind( psych.fname{ii},'LEDVALID')==1
        plot(eyelink.data{ii}(2,:),eyelink.data{ii}(3,:));
        plot(eyelink.data{ii}(4,:),eyelink.data{ii}(5,:),'g');
    end
end

for ii=1:4
    session = strcat(theSet{1},num2str(ii))
    for jj=1:length(psych.fname)
        if strcmp(psych.fname{jj},session)==1
            d=eyelink.data{jj}';
            [r,c]=size(d);
            data=[];
            for kk = 1:r
                if d(kk,6)==0 & d(kk,7)==0
                    data=[data;d(kk,:)];
                end
            end
            m = mean(data);
            pl(1,ii) = m(3);
            pl(2,ii) = m(2);
            pr(1,ii) = m(5);
            pr(2,ii) = m(4);
        end
    end
end