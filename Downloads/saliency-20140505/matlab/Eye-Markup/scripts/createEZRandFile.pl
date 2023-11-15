#!/usr/bin/perl 
#This script will take a bunch of e-ceyeS files and create a file of saccadic eye positions that can be used by ezvision instead of the standard uniform sampling of the saliency map during eyemovement comparision.This will create saccades for file x excluding eye-movements from file x  

use POSIX;

@TMPFILES = @ARGV;
foreach $file (@ARGV){
  print STDERR "\nProcessing $file\n";
  print STDERR "Writing Output File:$file-r\n";
  open(OUTFILE,">$file-r");
  foreach $file1 (@TMPFILES){
#lets cut the first part of the name off, so if it is the same cpit the if will fail
	my @tmpf = split(/\//,$file);
	my @tmpf1 = split(/\//,$file1);
	my $f = substr(pop(@tmpf),1,length($file));
	my $f1 = substr(pop(@tmpf1),1,length($file1));
	if ($f ne $f1){
	    #print "$f $f1\n"; 
	    open(INPFILE,"<$file1");
	    my $tmpl = <INPFILE>; #get rid of header lines
	    $tmpl = <INPFILE>;#get rid of header lines
	    $tmpl = <INPFILE>;#get rid od header lines
	    while(<INPFILE>){
		my $line = $_;
		chomp($line);
		#grab the x and y location
		my @tmp  = split(/ /,$line);
		#print STDERR "\n\n@tmp\n";
		@tmp = reverse(@tmp);
		my $tmpn = pop(@tmp);$tmpn = pop(@tmp);
		$tmpn = pop(@tmp);$tmpn = pop(@tmp);
		my $xp = floor(pop(@tmp));
		my $yp = floor(pop(@tmp));
		my $isSaccade = pop(@tmp);
		#if we have a saccade line lets add the endpoints 
		if (($isSaccade > 0) && ($xp > 0) && ($yp > 0) && ($xp < 640) && ($yp < 480)){
		    print OUTFILE "$xp $yp\n";
		    #  print STDERR "$isSaccade $xp $yp\n";
		}
	    }#end while
	  
	  close(INPFILE);
	}#end if file
    }#end foreach TMPFILES
  close(OUTFILE);
}#end foreach ARGV



