function saveCalibTxt(Cdata)
%function saveCalibTxt(Cdata)

for ii = 1:length(Cdata)
  [pathstr, fnam, ext, vers] = fileparts(Cdata(ii).fname);
  
  fnam = [fnam '.eye'];
  fil = fopen(fnam, 'w');
  if (fil == -1), disp(['Cannot write ' fnam]); return; end

  data = Cdata(ii).data;
  data(:,1:Cdata(ii).t) = [];
  
  data = [data; zeros(1, size(data,2))]; % add zero status everywhere
  % write some meta data here with ## header
  
  fprintf(fil, '%.1f %.1f %.1f %.1f\n', data);
  fclose(fil);
  disp(['Wrote ' fnam]);
end
