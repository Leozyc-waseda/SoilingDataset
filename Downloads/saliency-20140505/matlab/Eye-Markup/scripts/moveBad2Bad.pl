#!/usr/bin/perl
# Usage: moveBad2Bad.pl <badfile_list>
# The script move files in the badfiles (output of markBad.m) into
# "bad" folder under their directory
#
# Primary maintainer for this file: Po-He Tseng <ptseng@usc.edu>

$bad = $ARGV[0];
open(DAT, $bad);
@badlist = <DAT>;
close(DAT);

# prepare 'bad' directory
$badpath = substr($badlist[0], 0, rindex($badlist[0], '/'));
$badpath = "$badpath/bad";
&prep_dir($badpath);

foreach $badfile (@badlist){
	chomp($badfile);
	system("mv $badfile $badpath");
}

sub prep_dir
{
	$dir = $_[0];
	if (! -e $dir) {
		system("mkdir -p $dir");
		return;								  
	}
}

