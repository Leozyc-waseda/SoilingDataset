#!/usr/bin/perl

# USAGE:
# createZERandFile2.pl radius time number-agree </path/to/file1.eyesal> ... </path/to/fileN.eyesal>

#this code will take all the eyesal files in a directory and find the common saccades creating an eyesal-agree for each basename.  the structure of the name to your data file is assumed to be subj#-basename.eyesal


##########################################################################
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2007   ##
## by the University of Southern California (USC) and the iLab at USC.  ##
## See http://iLab.usc.edu for information about this project.          ##
##########################################################################
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; application number 09/912,225 filed July 23, 2001; see      ##
## http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     ##
##########################################################################
## This file is part of the iLab Neuromorphic Vision C++ Toolkit.       ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is free software; you can   ##
## redistribute it and/or modify it under the terms of the GNU General  ##
## Public License as published by the Free Software Foundation; either  ##
## version 2 of the License, or (at your option) any later version.     ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  ##
## that it will be useful, but WITHOUT ANY WARRANTY; without even the   ##
## implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ##
## PURPOSE.  See the GNU General Public License for more details.       ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   ##
## to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   ##
## Boston, MA 02111-1307 USA.                                           ##
##########################################################################
##
## Primary maintainer for this file: David Berg <dberg@usc.edu>

use strict;
use POSIX qw(floor);

my $rad = shift;
my $time = shift;
my $NumAgree = shift;
my %data; # saccades, indexed by eyesal file name
my $fbase = "";
my %usedlist;

#lets read the data from all the files and keep track of who 
#it belongs to
while ( my $f = shift) 
{
    my @tmpe = split(/\//,$f);
    pop(@tmpe);
    $fbase = join("/",@tmpe);
    #print (STDERR "Collecting data for: $f\n"); 
    local *F; open(F, "<$f") or die("Cannot open $f: ");
    while (<F>){
	my $line = $_;
        chomp($line);
        #the first column of our line is our index name
	my @tmp = split(/\s+/,$line);
	my $tmpname = shift(@tmp);
	$tmpname = removeNumber($tmpname);
	@tmp = split(/\./,$tmpname);
	$tmpname = shift(@tmp);
	print (STDERR "Collecting data for: $f\n"); 
	$data{$tmpname}.= "$line\n";
	$usedlist{$tmpname}.= "0\n";
    }
    close(F);
}

#ok now we have all the data, so lets just pass though each entry
foreach my $k (keys %data) {
    
    print STDERR "CALCULATING FOR $k\n";
    my $line = $data{$k}; 
    my @sacdata = split(/\n/,$line);
    my @ulist = split(/\n/,$usedlist{$k});
    
    #ok so this is all the data for one movie, lets see if there are any 
    #saccades to the same location and time
    #loop doubly through sacdata
    my $outline = "";
    my $ocount = 0;
    foreach my $ii (@sacdata){
	my @tmpii = split(/\s+/,$ii);
	my $ag_count = 1;
	my $icount = 0;
	
	foreach my $jj (@sacdata) {
	    my @tmpjj = split(/\s+/,$jj);
	    #do a little calculaton
	    
	    if (($tmpii[0] ne $tmpjj[0]) && ($ulist[$icount] != 1)){
		my $ex = $tmpii[1] - $tmpjj[1];
		my $ey = $tmpii[2] - $tmpjj[2]; 
		my $drad = sqrt(($ex*$ex) + ($ey*$ey));
		my $tdif = abs($tmpii[8] - $tmpjj[8]);
		if (($drad < $rad) && ($tdif < $time)) {
		    #$ulist[$ocount] = 1;
		    #$ulist[$icount] = 1;
		    $ag_count++; 
		    #$outline .= "$jj\n"
		}
	    }
	    $icount++;
	}

		
	if ($ag_count >= $NumAgree){	
	    $outline .= "$ii\n"
	    }
	else{
	    #$outline .= ""
	    }
	$ocount++;
    }    

    print STDERR "OUTPUT: ${fbase}/0-${k}.eyesal-agree\n";
    local *Fout; open Fout, ">${fbase}/0-${k}.eyesal-agree" || die " Cannot open ${k}=agree: ";
    print Fout $outline;
    close(Fout);

}#end loop through keys




sub removeNumber {
    my @tmp = split(/\//,$_[0]);
    my $tname = pop(@tmp);
    @tmp = split(/-/,$tname);
    shift(@tmp);
    $tname = join('-',@tmp);
    return $tname;
}
