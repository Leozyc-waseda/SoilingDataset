#!/usr/bin/perl

#
# This script converts a movie (possibly compressed with gzip or
# bzip2) into a .zip archive of BMP frames.
#
# USAGE: movie2bmp.pl [--slow] <movie1.avi> ... <movien.mov>
#
# will create the corresponding .zip files.
# CAUTION: will not work with grayscale of B/W movies!
#
# I have not figured out a way of avoiding frame drops if the machine
# is too slow to write out the frames at real play speed. As a
# temporary solution, the --slow flag will slow down playback by
# a factor 2. Use several --slow if one is not sufficient.

##########################################################################
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   ##
## by the University of Southern California (USC) and the iLab at USC.  ##
## See http://iLab.usc.edu for information about this project.          ##
##########################################################################
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; filed July 23, 2001, following provisional applications     ##
## No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).##
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
## $Id: movie2bmp.pl 10337 2008-10-13 17:24:09Z itti $
##

# select temp directory:
if (-d "/home/tmp/1") { $tdir = "/home/tmp/1/movie2bmp$$"; }
else { $tdir = "/tmp/movie2bmp$$"; }
print STDERR "### Using temporary directory $tdir\n";
if (! -d $tdir) { mkdir $tdir; }

# slow down playback if necessary to avoid dropping frames:
$speed = 1.0; while(@ARGV[0] eq '--slow') { $speed *= 0.5; shift; }
if ($speed < 1.0) { print STDERR "### Using speed $speed\n"; }

# commands to use:
$cmd_extract = "mplayer -vo png -nosound -z 1 -benchmark -nojoystick ".
    "-nolirc -cache 8192 -speed $speed";

foreach $f (@ARGV) {
    # make a temporary subdir for the frames of each movie:
    $tdir2 = "frames"; mkdir("$tdir/$tdir2");

    # uncompress if necessary...
    @ff = split(/\//, $f); $fbase = pop(@ff); $dir = join('/', @ff);
    @ff = split(/\./, $fbase); $ext = lc(pop(@ff)); $base = join('.', @ff);
    if ($ext eq 'bz2') {
        $avi = $fbase; print STDERR "Bunzipping $f...\n";
        system("/usr/bin/bunzip2 < $f > $tdir/$avi");
    } elsif ($ext eq 'gz') {
        $avi = $fbase; print STDERR "Gunzipping $f...\n";
        system("/bin/gunzip < $f > $tdir/$avi");
    } else {  # assume it will work...
        $avi = $fbase; print STDERR "Copying $f to temporary...\n";
        system("/bin/cp $f $tdir/$avi");
    }

    # the uncompressed movie is in $avi. Let's extract the frames:
    print STDERR "Extracing frames from $avi...\n";
    system("cd $tdir/$tdir2 && $cmd_extract ../$avi");

    # remove the movie file:
    system("/bin/rm $tdir/$avi");

    # get the new name for the .zip:
    @ff = split(/\./, $avi); pop(@ff); $base = join('.', @ff);

    # convert the frames: mplayer writes them out as XXXXXXXX.png starting
    # with frame 1, and we want frameXXXXXX.bmp instead starting at 0:
    # CAUTION: will not work with grayscale of B/W movies!
    print STDERR "Converting frames to BMP...\n";
    $ii = 1; $fr = sprintf("$tdir/$tdir2/%08d.png", $ii);
    while(-s $fr) {
        $newfr = sprintf("$tdir/$tdir2/frame%06d.bmp", $ii - 1);
        system("pngtopnm < $fr | ppmtobmp -windows -bpp=24 > $newfr");
        unlink($fr); $ii ++;
        $fr = sprintf("$tdir/$tdir2/%08d.png", $ii);
    }
    $ii --; print STDERR "$ii frames converted.\n";

    # pack the frames:
    print STDERR "Packing frames...\n";
    system("cd $tdir/$tdir2 && zip -9r ../$base.zip .");

    # move result back to current directory:
    print STDERR "Cleaning up...\n";
    if ($dir) {
        system("/bin/mv -f $tdir/$base.zip $dir/");
    } else {
        system("/bin/mv -f $tdir/$base.zip .");
    }

    # delete the frames:
    system("/bin/rm -rf $tdir/$tdir2");
}

# delete temp directory:
system("/bin/rm -rf $tdir");
