function resampleEyeS(glob,fdur,sampleto,fdursampleto,varargin)
tic
varout = parseinputs(varargin);
sf = getValue('Sf',varout);

p = round(sampleto*fdursampleto);
q = round(sf*fdur);


%get just the filenames
filz = dir([glob]);
for (ii = 1:length(filz))
    filez{ii} = filz(ii).name;
end

%a silly thing so the script works in windows and linux
f = findstr(glob,'\');
if (isempty(f))
    f = findstr(glob,'/');
end
glob(f(end)+1:end) = [];

%loop over our files
for ii = 1:length(filez)
    sdata = [];
    data = [];

    file = [glob,filez{ii}];

    disp(['Loading [' file ']....']);
    data = loadCalibTxt(file); %load the file
    %h = firls(length(data)*3,[0 .24 .34 .5],[1 1 0 0]);

    mb1 = mean(data(1,1:10));
    mb2 = mean(data(2,1:10));
    me1 = mean(data(1,end-9:end));
    me2 = mean(data(2,end-9:end));

    [sdata(1,:) b1] = resample(data(1,:)-mean([mb1 me1]),p,q,100);
    [sdata(2,:) b2] = resample(data(2,:)-mean([mb2 me2]),p,q,100);
    sdata(1,:) = sdata(1,:) + mean([mb1 me1]);
    sdata(2,:) = sdata(2,:) + mean([mb2 me2]);
    t1 = (0:size(data,2)-1)/sf;     % Time vector
    t2 = (0:(size(sdata,2)-1))*q/(p*sf);
    toc
    disp(['resampled ',file]);
    ext = 'reyeS';
    dot = find(file == '.');
    if (length(dot) > 0), file = file(1:dot(length(dot))); end
    file = [file, ext];
    fil = fopen(file, 'w');
    if (fil == -1), disp(['Cannot write ' file{ii}]); return; end
    fprintf(fil, '%.1f %.1f %d\n', [sdata',data(3,:)']);
    fclose(fil);
    disp(['Wrote ' file]);
end
