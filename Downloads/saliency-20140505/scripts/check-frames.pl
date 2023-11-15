#!/usr/bin/perl

# USAGE: check-frames.pl /path/to/framestem
#
# will check for a contiguous sequence of /path/to/framestemXXXXXX.anyext to
# /path/to/framestemYYYYYY.anyext and will report XXXXXX, YYYYYY, and any
# holes in the sequence.

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

my @flist = `/bin/ls $ARGV[0]*`;

my $start = -1; my $curr = -1; my $ret = 0; my $ext = "";
foreach my $f (@flist) {
    chomp $f; my @tmp = split(/\//, $f); my $fn = pop(@tmp);
    @tmp = split(/\./, $fn); my $base = shift(@tmp); my $e = join('.', @tmp);

    my $n = substr($base, -6);

    if ($n !~ /[0-9][0-9][0-9][0-9][0-9][0-9]/) {
	print STDERR "Invalid frame number at: $f\n"; $ret = 1;
    } else {
	my $nn = $n + 0;  # convert to int
	if ($start == -1) {
	    $start = $nn; $curr = $nn; $ext = $e;
	} elsif ($nn != $curr + 1) {
	    print STDERR sprintf("Missing frame between %06d and %06d\n", $curr, $nn); $ret = 2;
	} else { $curr ++; }

	if ($e ne $ext) { print STDERR "Inconsistent extension (not: .$ext) at $f\n"; $ret = 3; }
    }
}

if ($ret == 0) { print sprintf("$ARGV[0] %06d %06d\n", $start, $curr); }
else { print "$ARGV[0] ERROR ERROR\n"; }

exit $ret;
