#!/usr/bin/perl

# usage: parseISCAN.pl iscanfile.raw psychodata.psy >result.m

# will parse raw ISCAN eye tracker data and psycho-movie log file, and
# print out some results in matlab format. This will create a matlab
# m-file, which should be then run (just typing its name) in matlab;
# as it runs, it will save a .mat file with same name as the
# psychodata.psy file except for the .mat extension. Then you can
# delete the .m file and associated raw data, and just use the .mat
# file.

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
## Primary maintainer for this file: Laurent Itti <itti@usc.edu>
## $Id: COPYRIGHT-perl 4687 2005-06-11 14:31:37Z rjpeters $
##

use strict;

my $freq = 240; # sampling frequency

local *ISCAN; local *PSYCH;
open ISCAN, $ARGV[0] || die "Cannot open $ARGV[0]";
open PSYCH, $ARGV[1] || die "Cannot open $ARGV[1]";

# uncomment this to print debug messages:
sub dbg { print STDERR "$_[0]\n"; }  # show messages
#sub dbg { }                         # no messages

########## let's start by parsing the PSYCH file:
print STDERR "Parsing PSYCH...\n";
# first line is a timestamp: let's display it:
my $tim; my $line;
($tim, $line) = getpsych(); print STDERR "$line\n";
my $nn = 0; my $tracking = 0; my $starttim; my $movie;
my $fixx; my $fixy; my $image; my $image1; my $image2;
my @fx; my @fy; my @fname; my @toff; my @first; my @last;
my @comp1; my @comp2; my @comp3;
my $tp;
my @resptime; my @resp; my $latestresp; my  @objectlist;
my $lstrspt; my @objcount;

# loop over entire file:
while($tim > 0) {
    ($tim, $line) = getpsych();

    # start of an eye tracker session?
    if (substr($line, 0, 31) eq "----- Eye Tracker Start session") {
	$starttim = $tim; $tracking = 1;
	dbg("Eye-tracker start, session $nn");
    }

    # end of an eye tracker session?
    elsif (substr($line, 0, 30) eq "----- Eye Tracker Stop session") {
	dbg("Eye-tracker stop, session $nn");
	$tracking = 0; $nn ++;  # switch to next session
    }

    # start of an eye tracker calibration stimulus (a single flashing dot)?
    elsif (substr($line, 0, 31) eq "Begin: eyeTrackerCalibration at") {
	if ($tracking == 0) { die "Bogus psycho file\n"; }
	# let's push this new session into our data arrays and record
	# the coords of the calibration point as well as the time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	my @gogo = split(/[\(\), ]+/, $line);
	

	$fname[$nn] = "'CALIBRATION'"; $fx[$nn]=$gogo[3]; $fy[$nn]=$gogo[4];
	$comp1[$nn] = -1;$comp2[$nn] = -1;$comp3[$nn] = -1;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
	dbg("Begin eye-tracker calibration ($fx[$nn], $fy[$nn])");
    }

    # just loaded a movie to be played to the subject?
    elsif (substr($line, 0, 20) eq "===== Playing movie:") {
	# let's just keep track of the movie's name; once frame 0 of the
	# movie is displayed, we'll push all the data into our data
	# arrays:
	my @gogo = split(/ /, $line); $movie = $gogo[3];
	dbg("Playing movie $movie");
    }

    # first frame of a movie getting actually displayed?
    elsif (substr($line, 0, 27) eq "displayYUVoverlay - frame 0") {
	if ($tracking == 0) { die "Bogus psycho file\n"; }
	# let's push this new session into our data arrays and record
	# the name of the movie as well as the time elapsed (in eye
	# tracker ticks) since the eye tracker started:
	$fname[$nn] = "'$movie'"; $fx[$nn] = -1; $fy[$nn] = -1;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
	dbg("Got first frame of movie $movie");
    }


    # first frame of a movie getting actually displayed?
    elsif (substr($line, 0, 24) eq "MPlayerWrapper - frame 0") {
	if ($tracking == 0) { die "Bogus psycho file\n"; }
	# let's push this new session into our data arrays and record
	# the name of the movie as well as the time elapsed (in eye
	# tracker ticks) since the eye tracker started:
	$fname[$nn] = "'$movie'"; $fx[$nn] = -1; $fy[$nn] = -1;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
	dbg("Got first frame of mplayer movie $movie");
    }
    
    # displaying a fixation?
    elsif (substr($line, 0, 24) eq "Begin: displayFixation (") {
	# let's just keep track of the fixation coordinates in case we
	# need them later:
	my @gogo = split(/[\(\), ]+/, $line);
	$fixx = $gogo[2]; $fixy = $gogo[3];
    }

    # waiting for a key
     elsif (substr($line, 0, 17) eq "Begin: waitForKey") {
    	# let's just keep track of the waitkey in case we
        # need them later:
	$lstrspt = $tim;

     }	

    
    # waiting for a key 
    elsif ($line =~ m/waitForKey - got/) {
	# let's just keep track of the waitkey in case we 
	# need them later: 
	$lstrspt = $tim-$lstrspt;
	my @gogo = split(/ /,$line);
	$latestresp = $gogo[5];
    }    



    # displaying a static image (typically, a search array)?
    elsif (substr($line, 0, 28) eq "===== Showing search image: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image = $gogo[4];
	$fname[$nn] = "'$image'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
    }
    
    #displaying a text statement
    elsif (substr($line, 0, 24) eq "===== Showing Sentence: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	#if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image = $gogo[3];
	$fname[$nn] = "'sentence-$image'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
    } 

    #displaying a paragraph statement
    elsif (substr($line, 0, 25) eq "===== Showing paragraph: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	#if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image = $gogo[3];
	$fname[$nn] = "'$image'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
    } 

    #displaying a text statement
    elsif (substr($line, 0, 24) eq "===== Showing Question: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	#if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image = $gogo[3];
	$fname[$nn] = "'question-$image'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
	dbg("Got timestamp of question $image");
    } 


    #displaying a grid of text
    elsif (substr($line, 0, 28) eq "===== Showing Answer Grid: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started: 
	#if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image = $gogo[4];
	$fname[$nn] = "'$image'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
	# time at which stim appears, in eyetracker ticks:
	$toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
	dbg("Got timestamp of answer grid $image");
    } 

    
    # displaying a static image 
    elsif (substr($line, 0, 21) eq "===== Showing image: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	#if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image = $gogo[3];
	$fname[$nn] = "'$image'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
    }

    # displaying a static image 
    elsif (substr($line, 0, 22) eq "===== Showing images: ") {
	# let's push this new session into our data arrays and record
	# the name of the image, coordinates of last fixation and time
	# elapsed (in eye tracker ticks) since the eye tracker
	# started:
	#if ($tracking == 0) { die "Bogus psycho file\n"; }
	my @gogo = split(/ /, $line); $image1 = $gogo[3];$image2 = $gogo[5];
	my @timp = split(/\//,$image1);$image1 = pop(@timp);
	@timp = split(/\//,$image2);$image2 = pop(@timp);
	$fname[$nn] = "'$image1-$image2'"; $fx[$nn] = $fixx; $fy[$nn] = $fixy;
    }

    elsif (substr($line, 0, 21) eq "Begin: displayImage") {
	if ($tracking == 0) { 
	    dbg("Display No Tracking Image: $image"); 
	}
	else {	    
	    # time at which stim appears, in eyetracker ticks:
	    $toff[$nn] = int(($tim - $starttim) * $freq / 1000000.0);
	}
    } 

 elsif (substr($line, 0, 13) eq "===== Object ") {
     my @goto = split(/ /, $line);
     my $tmpstr = $goto[2];
     @goto = split(/\//,$tmpstr);
     $tmpstr = pop(@goto);
     push(@objectlist, "'$tmpstr'");
     push(@objcount,$nn-1);
     push(@resp, $latestresp);
     push(@resptime, $lstrspt/1000);
    } 



    elsif (substr($line, 0, 46) eq "===== Question: Which image is more beautiful?") {
	$tp = getQuestionResponse();
	$comp1[$nn-1] = $tp;
    } 

    elsif (substr($line, 0, 57) eq "===== Question: Which image will sell the product better?") {

	$tp = getQuestionResponse();
	$comp2[$nn-1] = $tp;
    }

    elsif (substr($line, 0, 46) eq "===== Question: Which image is 'the original'?") {

	$tp = getQuestionResponse();
	$comp3[$nn-1] = $tp;
	dbg("$comp1[$nn-1]\n$comp2[$nn-1]\n$comp3[$nn-1]\n");
    }

}

# ok, got all the data; write it out in matlab form:
print "clear global PSYCH;\n";
printlist('PSYCH.fname = ', \@fname);
printarray('PSYCH.fx = ', \@fx);
printarray('PSYCH.fy = ', \@fy);
printarray('PSYCH.t = ', \@toff);

if ($ARGV[1] =~ m/compare/)
{
    printarray('PSYCH.comp1 = ', \@comp1);
    printarray('PSYCH.comp2 = ', \@comp2);
    printarray('PSYCH.comp3 = ', \@comp3);

}

elsif ($ARGV[1] =~ m/memory/)
{
    printlist('PSYCH.objlist = ', \@objectlist);
    printarray('PSYCH.objrt = ', \@resptime);
    printarray('PSYCH.objresp = ', \@resp);
    printarray('PSYCH.objcount = ', \@objcount);
}

print STDERR "PSYCH: $nn sessions\n";

########## Now let's parse the ISCAN raw data file:
# get header line and check it:
print STDERR "Parsing ISCAN...\n";
$line = getiscan();
if ($line ne "ISCAN DATA FILE 2.06") { die "Not an ISCAN raw data file?\n"; }

# get subject, date, comment:
print "clear global ISCAN;\n";
$line = getiscan(); print "ISCAN.subject = '$line';\n";
$line = getiscan(); print "ISCAN.dat = '$line';\n";
$line = getiscan(); print "ISCAN.comment = '$line';\n";

# get number of sessions:
my $n = getiscan(); my $i = 0;
if ($n != $nn)
{ print STDERR "##### WARNING! $n ISCAN vs $nn PSYCH sessions! #####\n"; }

# parse session summaries; here we just care about the indices of the
# first and last records for each session, and about making sure that
# eye-tracker sampling rate was 240Hz:
while($i < $n) {
    $line = getiscan();
    my $x; my $hz; my $junk;
    ($first[$i], $last[$i], $x, $hz, $junk) = split(/\s+/, $line, 5);
    if ($hz != 240)
    { print STDERR "WARNING: session $i: sampling ${hz}Hz instead of 240Hz\n";}
    $i ++;
}

# parse parameter summaries; the goal here is to figure out which
# parameter keycode corresponds to each column in the recorded data:
my $done = 0; my $np = 1; my %pcode; # first is time stamp
while($done == 0) {
    $line = getiscan(); my @tmp = split(/\s+/, $line);
    # first element is the parameter code:
    if ($tmp[0] != 0) {
	$pcode{$tmp[0]} = $np++;
	# skip 3 lines of whatever:
	getiscan(); getiscan(); getiscan(); 
    } else { $done = 1; }
}
print STDERR "ISCAN: $n sessions, $np recorded variables\n";

# now find pupil H (code 1), V (code 2) and D (code 3),
# or POR-H (code 36), POR-V (code 37) and pupil D (code 3):

#my $idxH = $pcode{1};   # use Pupil-H as horiz eye pos
#my $idxV = $pcode{2};   # use Pupil-V as vertic eye pos
my $idxH = $pcode{36};  # we use POR-H as horiz eye pos
my $idxV = $pcode{37};  # we use POR-V as vertic eye pos
my $idxD = $pcode{3};   # pupil diameter
print STDERR "ISCAN columns: eyeH = $idxH, eyeV = $idxV, pupD = $idxD\n";
if ($idxH <= 0 || $idxV <= 0 || $idxD <= 0) { die "Missing some data!\n"; }

# when we get here, $line contains the first data point already; but
# somehow we have a leading zero on the first data line that we need
# to get rid of:
$line = substr($line, 2);

# let's loop over the sessions and get the H/V/D data that interests
# us. Because Matlab is a definite piece of trash, it will crash (no
# kidding!) if we just try to fill up the potentially long data arrays
# by just writing some 'array = [ value1 value2 ...];' code with, say
# 3,000 values (and breaking this array assignment nicely into short
# lines using the matlab '...' continuation markers does not help). So
# here instead we will write each session into a separate text file
# and in our matlab file we will write some matlab code to read those
# text files into matlab arrays using fscanf. The cause for this
# headache is that matlab just sucks!
$i = 0;  # session number (0 .. $n)
do {
    # here we have the first data line of a given session to be read
    # in stored into $line. Let's start by breaking the data line down
    # into fields:
    my @tmp = split(/\s+/, $line);

    # first field is the sample number; should match what the
    # summaries told us the first sample number should be:
    if ($tmp[0] != $first[$i]) { die "Bogus session start at $tmp[0]\n"; }

    # ok, let's read the rest of this session, get only the fields we
    # want, and write it back into a separate text file for each
    # session (to be later read in by matlab):
    my $ii = $i + 1;   # 1-based session number
    my $fnam = $ARGV[0].'.'.sprintf('%04d', $ii);
    local *F;
    open F, ">$fnam" || die "Cannot write $fnam";
    my $npts = $last[$i] - $first[$i] + 1; my $j = 1; # already have first line
    do {
	print F "$tmp[$idxH] $tmp[$idxV] $tmp[$idxD]\n";
	$j ++; $line = getiscan(); @tmp = split(/\s+/, $line);
    } while($j < $npts);
    close F;
    print "f = fopen('$fnam');\n";
    print "ISCAN.data{$ii} = fscanf(f, '%f', [3, Inf]); fclose(f);\n";

    $i ++; # next session
    print STDERR "Parsed session $i/$n...\n";
} while ($line = getiscan());

close ISCAN; close PSYCH;

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
sub getiscan { 
    my $line = <ISCAN>;
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


######################################################################
sub getQuestionResponse {
    my $tim; my $line; my $resp;   
    ($tim, $line) = getpsych(); 
    while (substr($line, 0, 23) ne "End : waitForKey - got ") {
	($tim, $line) = getpsych();	
    }
    my @goto = split(/ /,$line);
    $resp = int($goto[5]);
    return ($resp);
}
