function processdir(glob)
%if you cant use the GUI then try this function:  Just change the
%markeye code below. This is an example of batching several different 
%parsings of an eye trace file.  
filz = dir([glob]);
for (ii = 1:length(filz))
    filez{ii} = filz(ii).name;
end
%a silly thing so the script works in windows and linux
f = findstr(glob,'\');
if (isempty(f))
    f = findstr(glob,'/');
end
glob(f(end)+1:end) = []
%a silly thing so the script works in windows and linux


%edit below here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for (aa = [30]); %all our all thresholds    
    for ( jj = [100] )%all our timethreshold for prosaccades
        
        %loop over our files (you'll need at least this)
        for ii = 1:length(filez)
            fnam = filez{ii};
            fnam = [glob,fnam];
            
           %call markeye and change to your liking
            markEye(fnam,'sf',1000,'ppd',[10 10],'sacpca-window',...
                [5 10],'sac-pcathresh',.01,'autosave',1,'pro-anglethresh',aa,...
                'pro-timethresh',jj,'out-ext',[num2str(jj),'-',num2str(aa),...
                '-','ceyeS']);
        end%end over files (for minimal batching without gui use this)
        
    end%end jj
end% end aa

