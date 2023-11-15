#!/usr/bin/perl

# USAGE: movieFromStills.pl <still1.ppm> ... <stillN.ppm>

# Will create a .tbz movie called movie.tbz in which each still image
# is shown for a few frames, then faded out, then on to the next
# still, etc.  See hardcoded params at the start of this program's
# source for adjustments.

# This requires saliency/bin/fade_image to be in the path

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
## $Id: movieFromStills.pl 10311 2008-10-01 22:14:04Z itti $
##

$nshow = 20;  # number of frames during which each still image is shown
$nfade = 5;   # number of frames over which each still image is faded out
$cmd_show = 'fade_image none';  # command to replicate the image
$cmd_fade = 'fade_image out';   # command to fade the image out

######################################################################
$frame = 0;
# select temp directory:
if (-d "/home/tmp/1") { $tdir = "/home/tmp/1/movieFromStills$$"; }
else { $tdir = "/tmp/movieFromStills$$"; }
print STDERR "### Using temporary directory $tdir\n";
if (! -d $tdir) { mkdir $tdir; }

# remember where we started from:
$dest = `pwd`; chomp $dest;

# copy the stills over:
print STDERR "### Copying still images to temporary\n";
foreach $f (@ARGV) { system("/bin/cp $f $tdir/$f"); }

# get ready:
mkdir("$tdir/movie"); chdir("$tdir/movie");
foreach $f (@ARGV) {
    # eliminate old path:
    @ff = split(/\//, $f); $f = pop(@ff);

    # show the still image:
    $start = $frame; $end = $frame + $nshow - 1; $frame += $nshow;
    print STDERR "### Showing $f, frames $start -- $end\n";
    system("$cmd_show ../$f $start $end");

    # show the still image:
    $start = $frame; $end = $frame + $nfade - 1; $frame += $nfade;
    print STDERR "### Fading $f, frames $start -- $end\n";
    system("$cmd_fade ../$f $start $end");
}

# create .tbz:
print STDERR "### Compressing movie...\n";
system("tar cf - . | bzip2 -9 > $dest/movie.tbz");

# cleanup:
chdir($dest);
print STDERR "### Cleaning up...\n";
system("/bin/rm -rf $tdir");
