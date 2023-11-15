function xy = loadCalibTxt(fname)
%load an eye or eyeS file and return it in a variables by time matrix
%with three columns [x y pd], pd = pupil diameter.
%xy is a 3xN matrix, pupil is a 1xN matrix 

% should load comments here
ferp = fopen(fname,'r');
R = 0;
w = fgetl(ferp);
while ( ~feof(ferp) && isnan(str2double(w(1))) )
  w = fgetl(ferp);
  R = R + 1;
end
fclose(ferp);

xy = dlmread(fname,' ',R,0)';
r = size(xy,1);
pupil = zeros(1,length(xy));
if (r == 2)
  xy = [xy;zeros(1,length(xy))];
else
  xy = xy(1:3,:);
%{
elseif (r ==3)

  pupil = xy(3,:);
  xy = [xy(1:2,:);zeros(1,length(xy))];
elseif ((r == 4) || (r == 8)  || (r == 9))
  pupil = xy(3,:);
  xy = xy([1 2 4],:);
elseif (r == 7)
  xy = xy(1:3,:);
%}
end



