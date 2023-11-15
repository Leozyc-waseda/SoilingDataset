#!/usr/bin/perl
# USAGE: add-copyright.pl <copyright-file> <source-file>
# will insert copyright file at the first blank line of source-file

## #################################################################### ##
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the ##
## University of Southern California (USC) and the iLab at USC.         ##
## See http://iLab.usc.edu for information about this project.          ##
## #################################################################### ##
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; filed July 23, 2001, following provisional applications     ##
## No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).##
## #################################################################### ##
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
## #################################################################### ##
##
## Primary maintainer for this file: Laurent Itti <itti@usc.edu>
## $Id: add-copyright.pl 4302 2005-06-04 17:01:21Z itti $
##

open F, $ARGV[0] || die "Cannot open $ARGV[0]";
@cop = <F>; close F;

system("/bin/mv $ARGV[1] $ARGV[1].old");

open F, "$ARGV[1].old" || die "Cannot open $ARGV[1].old";
open D, ">$ARGV[1]" || die "Cannot write $ARGV[1]";
print STDERR "Inserting copyright '$ARGV[0]' into '$ARGV[1]'.\n";
$todo = 1;
while($line = <F>) {
    if ($todo) {
	$test = $line; $test =~ s/\s+//g;
	if ($test eq "") {
	    print D "\n"; print D @cop; print D "\n";
	    $todo = 0;
	} else { print D $line; }
    } else { print D $line; }
}
close F; close D;
