function result = idRegions(data,params, gons)
%uses VideoLabelMe data to label eyetrace according to regions of interest
% 
% adds a field to the data about which polygons are found  mean what
% may be added as comment to e-ceyeS,conf file, or by class/dictionary here
% does not handle NaNs from blinks very well as of now
%written: Oct 2009
%author: John Shen

% iLab - University of Southern California
%**************************************************************************

result = data;
if ~exist('gons') || isempty(gons)
  return;
end

dict = getvalue('label_dict',params);
sf = getvalue('sf',params);
label_sf = getvalue('sf',params);

% default region is 0 for none
result.region = zeros(1,data.len);

% see which polygons claim a POR
inside = zeros(data.len,length(gons));
for jj = 1:length(gons)
  % interpolate moving polygons - time in msec
  t_frame = single(gons(jj).polygon.t)/label_sf*1000;
  t_samples = (0:data.len-1)/sf*1000;

  if gons(jj).moving
    gons(jj).polygon.x = transpose(interp1(t_frame,...
					   gons(jj).polygon.x',...
					   t_samples, 'linear','extrap'));
    gons(jj).polygon.y = transpose(interp1(t_frame,...
					   gons(jj).polygon.y',...
					   t_samples, 'linear','extrap'));
    gons(jj).polygon.t = t_samples;
    for kk = 1:data.len
      % this not only finds regions but also resolve conflicts w/ priority
      % right now resolution is just by what is first labeled
      % this takes advantage of the fact that eyes,face,mouth are
      % usually done first
      if any(inside(kk,1:jj-1)) continue; end;
      gon = [gons(jj).polygon.x(:,kk), gons(jj).polygon.y(:,kk)];
      inside(kk,jj) = inpoly(data.xy(:,kk)',gon);
    end
  else
    unassigned = find(~any(inside(:,1:jj-1),2));
    gon = [gons(jj).polygon.x(:,1), gons(jj).polygon.y(:,1)];
    inside(unassigned,jj) = inpoly(data.xy(:,unassigned)',gon);
  end
end

% write map from VLM labels to regions
VLM_to_region = zeros(1,length(gons));
for jj = 1:length(gons)
  VLMname = gons(jj).name;
  for kk = 1:length(dict)
    defn = dict{kk};
    for ll = 1:length(defn)
      entry = char(defn{ll});
      if strfind(VLMname, entry)
	VLM_to_region(jj) = kk;
	break;
      end
    end
    if VLM_to_region(jj)>0, break; end;
  end
end
VLM_to_region(VLM_to_region==0) = length(dict); %default is last

for ii = 1:data.len
  % add labels
  if any(inside(ii,:))
    result.region(ii) = VLM_to_region(find(inside(ii,:),1,'first'));    
  else
    result.region(ii) = length(dict); % default is last
  end

end

%{
probe = [1 2 3 7];
subplot(6,1,1:2)
hold on;
for j = 1:length(probe)
  jj = probe(j);
  polyx = gons(jj).polygon.x(:,1);
  polyy = gons(jj).polygon.y(:,1);
  polyx(end+1)=polyx(1);
  polyy(end+1)=polyy(1);
  plot(polyx,polyy,'r-')
  
  idx_huh = find(result.region==jj)
  result.xy(:,idx_huh)
  colstyle = {'m.','r.','y.','b.'};
  if j <= length(probe)
    plot(result.xy(1,idx_huh),result.xy(2,idx_huh),colstyle{j},...
	 'MarkerSize', 15);
  end
end
pause
%}
%{
% apply booleans based on labels
for kk = 1:length(dict)
  entries = dict{kk};
  keyentry = entries{1};
  result.(['at' keyentry]) = strcmp({gons(result.region).label}, keyentry);
end


 %}