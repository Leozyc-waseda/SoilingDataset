#!/usr/bin/perl

# USAGE:
# createZERandFile2.pl </path/to/file1.e-ceyeS> ... </path/to/fileN.e-ceyeS>
#
# Will read all the e-ceyeS files, and will extract the eye coordinates of all
# human (or monkey) saccade targets. Then will create one e-ceyeS-r file
# for each e-ceyeS file, which contains a list of all target coordinates
# except for those which came from the same stimulus (matching e-ceyeS name
# even though the paths may be different).
#
# This is used by ezvision as a prior distribution for selecting random
# samples. Here the samples will be selected from all saccades except the
# ones that were made while watching the same stimulus as the one
# being tested with ezvision.

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
## Primary maintainer for this file: Laurent Itti <itti@usc.edu>
## $Id: COPYRIGHT-perl 4687 2005-06-11 14:31:37Z rjpeters $
##


use strict;
use POSIX qw(floor);

my $w = 640;
my $h = 480;

print STDERR "Assumed display size: ${w}x${h}\n";

my %data; # saccades, indexed by e-ceyeS name

# first, let's gobble in all the saccades, each time keeping track of
# which stimulus they came from:
print STDERR "Gathering saccades:";
foreach my $f (@ARGV) {
    print STDERR " $f";
    local *F; open F, $f || die "Cannot open $f: ";
    my @ff = split(/\//, $f); my $fname = pop(@ff);

    while(<F>) {
	if (m/[=\#]/) { next; } # skip comments and metadata
	chomp; my @x = split(/\s+/);

	# see saliency/src/Psycho/EyeTrace.C to make sure the format
	# has not changed!
	if ($#x != 7) { die "Incorrect input format (need 8 fields). "; }

	# got a saccade start (ampl > 0)?
	if ($x[6] > 0.0) {
	    # get integer saccade target coords:
	    my $xx = floor($x[4] + 0.49999);
	    my $yy = floor($x[5] + 0.49999);
	    if ($xx >= 0 && $xx < $w && $yy >= 0 && $yy < $h) {
		$data{$fname} .= "$xx $yy\n";
	    }
	}
    }
    close(F);
}
print STDERR " - DONE.\n\n";

# all right, now create the output files:
print STDERR "Creating output files:\n";
foreach my $f (@ARGV) {
    my @ff = split(/\//, $f); my $fname = pop(@ff);
    print STDERR "... ${f}-r\n";
    local *F; open F, ">${f}-r" || die " Cannot open ${f}-r: ";

    foreach my $k (keys %data) {
	# spit out the saccades from all stimuli except the one under test:
	if ($k ne $fname) { print F $data{$k}; }
    }
    close(F);
}
