function modelfree(glob,varargin)
%this function will calculate and save model free a model free stats
%file for each e-ceyeS file, getting directory names from
%datalist.txt.  It save angle of saccade, amplitutde,
%peak veloctiy, intersaccdic interval, duration of fixation before,
%%duration of fixation after if no event exists (ie, no fixation
%before saccade) then a 0 is put in the file.

addpath('~/saliency/matlab/Eye-Markup');


[name,glob] = strip_file_path(glob);
for (jj = 1:length(name))
  fname = [glob,name{jj}];
  disp(['Loading file:', fname]);
  clear amp pvel angle intersac bdur adur;
  [amp,pvel,angle,intersac,bdur,adur] = modelfreestats(fname,varargin);
  fnameo =[fname,'.mfree'];
  fil = fopen(fnameo,'w');
  if (fil == -1);error(['Cannot open ',fnameo]);end
  disp(['Writing:', fnameo]);
  for (kk = 1:length(amp))
    fprintf(fil,['%3.2f %3.2f %3.2f %3.2f %3.2f ' '%3.2f\n'], ...
            angle(kk),amp(kk),pvel(kk),intersac(kk), bdur(kk),adur(kk));
  end
  fclose(fil);
end