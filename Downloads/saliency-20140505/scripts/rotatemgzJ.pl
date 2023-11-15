#!/usr/bin/perl
# rotateMgzJ.pl <mgzJ file> [-90|90|180] 
# clockwise is positive direction
# will take nonmultiples of 90 as well

$movieFile = shift;
$rotateAngle = shift; # clockwise is positive

if (!($movieFile =~ m/\.mgzJ$/) || ! -e $movieFile) 
  {die("Not an mgzJ file\n");}

# create temporary folder
$tmpFolder = "/tmp/movie_".join('_', localtime(time));
system("mkdir -p $tmpFolder");

# unpack into temporary folder
$streamProg = "bin/stream";
$cmd="$streamProg --in=$movieFile --input-frames=0-MAX\@30Hz".
    " --out=pnm:${tmpFolder}/ --output-frames=0-MAX\@30Hz";
system($cmd);

# do the rotation with mogrify
system("mogrify -rotate ${rotateAngle} ${tmpFolder}/stream-output\*.pnm");

(my $newFile = $movieFile) =~ s/\.mgzJ$/-rotated\.mgzJ/;
(my $movieStreamFile = $newFile) =~ s/rotated/rotatedstream-output/;

# repack into new, rotated file
print("Writing new movie to $movieStreamFile...\n");

$cmd="$streamProg --in=raster:${tmpFolder}/stream-output#.pnm --input-frames=0-MAX\@30Hz".
    " --out=mgzJ:${newFile}/ --output-frames=0-MAX\@30Hz";
system($cmd);

# clear old directory
# make sure tmpFolder is still part of tmp
if($tmpFolder =~ m/^\/tmp/) 
  {system("rm -rf ${tmpFolder}");}

print("Overwriting $movieFile...\n");
system("mv $movieStreamFile $movieFile");

