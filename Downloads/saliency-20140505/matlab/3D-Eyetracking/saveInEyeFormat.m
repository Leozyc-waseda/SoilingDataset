function saveInEyeFormat(name,f,type)



load(name);

if nargin==2
    for ii =1:length(EYELINK.data)
        fnam=[PSYCH.answer{ii} '_' PSYCH.fname{ii}   '.aeye'];
        fil = fopen(fnam, 'w');
        if (fil == -1), disp(['Cannot write ' fnam]); return; end
        d = EYELINK.data{ii};
        [r c] = size(d);
        data =zeros(3,c);
        data(3,:)=d(1,:);
        data(1,:)=(d(2,:)+d(4,:))/2;
        data(2,:)=(d(3,:)+d(5,:))/2;
        od = fillTheGapInEyeLinkData(data,f);
        od =[od;zeros(1,length(od))];
        fprintf(fil, '%.1f %.1f %d %.1f\n', od);
        fclose(fil);
        disp(['Wrote ' fnam]);
        end
end