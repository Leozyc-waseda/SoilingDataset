#!/usr/bin/perl

# USAGE: cripple_doc.pl <path>
# Looks for files of the form:
#
#    *_8<ext>-source.html
#
# with <ext> in { C, cpp, cc, c, java, Q } under <path>, deletes them, and
# replaces them by symlinks to ../../nosrc.html (note how this assumes
# that all files will be two directories down from <path> as is the case
# with doxygen docs when subdirectories are turned on).
#
# This is used to cripple a doxygen-generated documentation so that
# everything except the C/C++/java implementation source code is
# available (class definitions are still available).

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

# get the path:
$p = shift || die "USAGE: $0 <path>";

# extentions to cripple:
@ext = qw/ C cpp cc c java Q cu /;

# let's do it:
foreach $x (@ext) {
    @files = `/usr/bin/find $p -name "*_8${x}-source.html" -print`;
    foreach $f (@files) {
	chomp $f;
	print STDERR "Crippling: $f\n";
	unlink($f) || die "Cannot unlink $f: $!";
	symlink("../../nosrc.html", $f) || die "Cannot symlink $f: $!";
    }
}
