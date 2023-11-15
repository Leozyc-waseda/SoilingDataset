function objdata = setObjectData(VLM_data, params, vidfile)
% gets the data for a labelme annotation when a directory is saved
% given the annotation struct array and the file, 
% also applies shorter labels to the regions
% and scales the regions up in space
% returns empty struct if there is no annotation
dict = getvalue('label_dict',params);
vblevel = getvalue('verbose',params);

annot_dims = getvalue('label_dims',params);
screen_dims = getvalue('screen_size',params);
objdata = struct([]);

% isolate polygon data, if it exists
for ii = 1:length(VLM_data)
  if strcmp(VLM_data(ii).annotation.filename,vidfile)
    objdata = VLM_data(ii).annotation.object;

    if vblevel>=1
      fprintf('Found annotations for %s...\n', vidfile);
    end
   
    break;
  end
end
  
%objdata(1).label = zeros(1,length(objdata));
% give shorter region #s to object categories
for jj = 1:length(objdata)
  assigned = 0;
  for kk = 1:length(dict)-1
    entries = dict{kk};
    for ll = 1:length(entries)
      this_entry = entries(ll);
      if strfind(objdata(jj).name,this_entry{:}) 
	assigned = 1;
	objdata(jj).label=entries{1};
	break;
      end
    end    
    if assigned, break; end;
  end
  if ~assigned, objdata(jj).label = dict{end}; end;
end

% preprocessing in space
mult = screen_dims./annot_dims;
for jj = 1:length(objdata)
  % scale up polygons
  objdata(jj).polygon.x = objdata(jj).polygon.x * mult(1);
  % invert in y direction
  objdata(jj).polygon.y = screen_dims(2) - objdata(jj).polygon.y * mult(2);
end
