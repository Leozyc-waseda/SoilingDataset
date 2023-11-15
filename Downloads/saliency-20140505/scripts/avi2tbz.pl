#!/usr/bin/perl

#
# This script converts an AVI movie (possibly compressed with gzip or
# bzip2) into a .tbz archive of raw PPM frames for use, for example,
# with process_movie.pl. The AVI codec must be supported by our
# xanim+ppm program (located in the beobots CVS repository).
#
# USAGE: avi2tbz.pl <movie1.avi> ... <movien.avi>
#
# will create the corresponding .tbz files.
#

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
## $Id: avi2tbz.pl 10337 2008-10-13 17:24:09Z itti $
##

# commands to use:
$cmd_extract = "xanim+ppm +ppm +N +Ze";

# select temp directory:
if (-d "/home/tmp/1") { $tdir = "/home/tmp/1/avi2tbz$$"; }
else { $tdir = "/tmp/avi2tbz$$"; }
print STDERR "### Using temporary directory $tdir\n";
if (! -d $tdir) { mkdir $tdir; }

foreach $f (@ARGV) {
    # make a temporary subdir for the frames of each movie:
    $tdir2 = "frames"; mkdir("$tdir/$tdir2");

    # uncompress if necessary...
    @ff = split(/\//, $f); $fbase = pop(@ff); $dir = join('/', @ff);
    @ff = split(/\./, $fbase); $ext = lc(pop(@ff)); $base = join('.', @ff);
    if ($ext eq 'bz2') {
        $avi = $base; print STDERR "Bunzipping $f...\n";
        system("/usr/bin/bunzip2 < $f > $tdir/$avi");
    } elsif ($ext eq 'gz') {
        $avi = $base; print STDERR "Gunzipping $f...\n";
        system("/bin/gunzip < $f > $tdir/$avi");
    } elsif ($ext eq 'avi') {
        $avi = $f; print STDERR "Copying $f to temporary...\n";
        system("/bin/cp $f $tdir/$avi");
    } else {
        die "Unknown file format for $f\n";
    }

    # the uncompressed movie is in $avi. Let's extract the frames:
    print STDERR "Extracing frames from $avi...\n";
    system("cd $tdir/$tdir2 && $cmd_extract ../$avi");

    # remove the movie file:
    system("/bin/rm $tdir/$avi");

    # get the new name for the .tbz:
    @ff = split(/\./, $avi); pop(@ff); $base = join('.', @ff);

    # pack the frames:
    print STDERR "Packing frames...\n";
    system("cd $tdir/$tdir2 && /bin/tar cf - . | /usr/bin/bzip2 -9 ".
           "> ../$base.tbz");

    # move result back to current directory:
    print STDERR "Cleaning up...\n";
    if ($dir) {
        system("/bin/mv -f $tdir/$base.tbz $dir/");
    } else {
        system("/bin/mv -f $tdir/$base.tbz .");
    }

    # delete the frames:
    system("/bin/rm -rf $tdir/$tdir2");
}

# delete temp directory:
system("/bin/rm -rf $tdir");
