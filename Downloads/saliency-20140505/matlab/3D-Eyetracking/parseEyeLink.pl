#!/usr/bin/perl

# usage: parseEyeLink.pl eye-link.asc psychodata.psy >result.m

# will parse raw EyeLink eye tracker data and psycho log file, and
# print out some results in matlab format. This will create a matlab
# m-file, which should be then run (just typing its name) in matlab;
# as it runs, it will save a .mat file with same name as the
# psychodata.psy file except for the .mat extension. Then you can
# delete the .m file and associated raw data, and just use the .mat
# file.
# in order to have a happy parsing session you need to have your recording
# done in binocular setting and the result should be presented in screen calibrated 
# format. 

##########################################################################
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   ##
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
## Primary maintainer for this file: Nader Noori <nnoori@usc.edu>
##

use strict;

local *EYELINK; local *PSYCH;
open EYELINK, $ARGV[0] || die "Cannot open $ARGV[0]";
open PSYCH, $ARGV[1] || die "Cannot open $ARGV[1]";

# uncomment this to print debug messages:
#sub dbg { print STDERR "$_[0]\n"; }  # show messages
sub dbg { }                         # no messages

########## let's start by parsing the PSYCH file:
print STDERR "Parsing PSYCH...\n";
# first line is a timestamp: let's display it:
my $tim; my $line;
($tim, $line) = getpsych(); print STDERR "$line\n";
my $nn = 0; my $tracking = 0; 
my @fname; my @toff; my @quality ;
my $tp;

# loop over entire psycho log file:
while($tim > 0) {
    ($tim, $line) = getpsych();

    # start of an eye tracker session?
    if (substr($line, 0, 18) eq "displayEvent start") {
	
	    my @gogo = split(/[\(\), ]+/, $line);
	    my $tmpstr=$gogo[2];

        if($tmpstr eq "STIMULUS"){
            $tmpstr=$gogo[4];
            $fname[$nn]="'$tmpstr'";
            $tracking = 1;
            
        }

	
	    dbg("A display event started, session $nn , event $tmpstr ");
    }

    # end of an eye tracker session?
    elsif (substr($line, 0, 26) eq "displayEvent stop STIMULUS") {
	dbg("Eye-tracker stop, session $nn");
	    $tracking = 0; 
    }
    
    # we will check whether the answer was correct
    if(substr($line, 0, 18) eq "answer was correct"){
        $quality[$nn]="'g'";
        $nn ++;         
    }elsif(substr($line, 0, 20) eq "answer was incorrect"){
        $quality[$nn]="'b'";
        $nn ++; 
    }
    
}

# ok, got all the data; write it out in matlab form:
print "clear global PSYCH;\n";
printlist('PSYCH.fname = ', \@fname);
printlist('PSYCH.answer = ', \@quality);


print STDERR "PSYCH: $nn sessions\n";

########## Now let's parse the EYELINK raw data file:
# get header line and check it:
print STDERR "Parsing EYELINK...\n";
# let's skip three lines and see whether the fourth line matches the thing we expect!
$line = geteyelink();$line = geteyelink();$line = geteyelink();$line = geteyelink();
if ($line ne "** VERSION: EYELINK II 1") { die "Not an EyeLink raw data file?\n"; }


print "clear global EYELINK;\n";


my $i = 0;
my $active=0 ;
my $lEvent=0;
my $rEvent=0;
my $lEvent=0;
while($i < $nn) {
    $line = geteyelink();

    if($active == 0 & (substr($line,0,5) eq "START")){
        $active = 1 ;
        my $ii = $i + 1;   # 1-based session number
        print STDERR "Parsing session ... $ii/$nn...\n";

        my $fnam = $ARGV[0].'.'.sprintf('%04d', $ii);
        print "f = fopen('$fnam');\n";
        print "EYELINK.data{$ii} = fscanf(f, '%f', [7, Inf]); fclose(f);\n";
        local *F;
        open F, ">$fnam" || die "Cannot write $fnam";
        
        while ($active == 1 & (substr($line,0,3) ne "END")){
            $line = geteyelink();
            my @tmp = split(/\s+/, $line);
            if(($tmp[0] eq "SFIX") && ($tmp[1] eq "L") ){ $lEvent = 0 ;}
            if($tmp[0] eq "SFIX" && $tmp[1] eq "R" ) {$rEvent = 0 ;}
            if($tmp[0] eq "SSACC" && $tmp[1] eq "L" ) {$lEvent = 1 ;}
            if($tmp[0] eq "SSACC" && $tmp[1] eq "R" ) {$rEvent = 1 ;}
            
            if($tmp[7] eq "....."){
                 print F "$tmp[0] $tmp[1] $tmp[2] $tmp[4] $tmp[5] $lEvent $rEvent\n";
            }
            
        }

        print STDERR "Parsed session  $ii/$nn\n";
        $i++ ;
        $active = 0 ;
        close F;
    }


}


close EYELINK; close PSYCH;

# remove the extension from the psych data file name:
my @tmp = split(/\./, $ARGV[1]); pop(@tmp); my $base = join('.', @tmp);

# add a command so that matlab saves the data as 'psychname.mat':
print "clear ans; clear f; save '$base.mat';\n";

######################################################################
sub printarray { # name, array ref
    my $name = shift; my $array = shift;
    print "$name\[ "; my $n = 0; my $val;
    while($val = shift(@$array)) {
	print "$val "; $n++;
	if ($n % 10 == 0) { print "...\n"; }
    }
    print "];\n";
}

######################################################################
sub printlist { # name, array ref
    my $name = shift; my $array = shift;
    print "$name\{ "; my $n = 0; my $val;
    while($val = shift(@$array)) {
	print "$val "; $n++;
	if ($n % 10 == 0) { print "...\n"; }
    }
    print "};\n";
}

######################################################################
sub geteyelink { 
    my $line = <EYELINK>;
    $line =~ s/\s+/ /g; $line =~ s/^\s+//; $line =~ s/\s+$//;
    return $line;
}

######################################################################
sub getpsych {
    my $line = <PSYCH>;
    $line =~ s/\s+/ /g; $line =~ s/^\s+//; $line =~ s/\s+$//;
    my $tt; my $rest; ($tt, $rest) = split(/ /, $line, 2);
    my @t = split(/[:\.]/, $tt);
    my $tim = $t[0] * 3600000000.0 + $t[1] * 60000000.0 + $t[2] * 1000000.0 +
	$t[3] * 1000.0 + $t[4];  # time in usec
    return ($tim, $rest);
}


