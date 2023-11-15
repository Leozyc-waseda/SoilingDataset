#!/usr/bin/perl

#
# This script converts a movie (possibly compressed with gzip or
# bzip2) into a .mpg MPEG-1 movie.
#
# This version uses the mplayer program to do the frame extraction
# from movie files. It may advantageously replace avi2tbz.pl which
# relies on the (now very old) xanim+ppm. Should support many types of
# files and codecs (AVI but also quicktime, DIVX, etc)
#
# USAGE: movie2mpg.pl [--slow] [--hq] [--fbeg=frame] [--fend=frame]
#                     <movie1.avi> ... <movien.mov>
#
# will create the corresponding .mpg files.
# CAUTION: will not work with grayscale or B/W movies!
#
# If --fbeg and/or --fend are specified, only encode that range of
# frames (both bounds inclusive) into the .tbz; otherwise take all frames.
#
# I have not figured out a way of avoiding frame drops if the machine
# is too slow to write out the frames at real play speed. As a
# temporary solution, the --slow flag will slow down playback by a
# factor 2. Use several --slow if one is not sufficient. If --hq is
# specified, a high-quality MPEG-1 encoder setting will be used.

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
## $Id: movie2tbz.pl 3872 2004-10-22 21:11:36Z itti $
##

# select temp directory:
if (-d "/home/tmp/1") { $tdir = "/home/tmp/1/movie2tbz$$"; }
else { $tdir = "/tmp/movie2tbz$$"; }
print STDERR "### Using temporary directory $tdir\n";
if (! -d $tdir) { mkdir $tdir; }

# parse command line:
$speed = 1.0; $do_mpghq = 0; $fbeg = 0; $fend = 2000000000;
foreach $a (@ARGV) {
    if ($a eq '--slow') { $speed *= 0.5; }
    elsif ($a eq '--hq') { $do_mpghq = 1; }
    elsif (substr($a, 0, 6) eq '--fbeg')
    { @x = split(/=/, $a); $fbeg = $x[1]; }
    elsif (substr($a, 0, 6) eq '--fend')
    { @x = split(/=/, $a); $fend = $x[1]; }
}

if ($speed < 1.0) { print STDERR "### Using speed $speed\n"; }
$range = '';
if ($fend < 2000000000) { $range = "$fbeg-$fend "; }
elsif ($fbeg > 0) { $range = "$fbeg-END "; }
if ($range) { print STDERR "### Using frame range: $range\n"; }

# commands to use:
$cmd_extract = "mplayer -vo png -nosound -benchmark -nojoystick ".
    "-nolirc -cache 8192 -speed $speed";
$cmd_conv = "pngtopnm";

foreach $f (@ARGV) {
    # skip command-line options:
    if (substr($f, 0, 2) eq '--') { next; }

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

    # get the new name for the .tbz:
    @ff = split(/\./, $avi); pop(@ff); $base = join('.', @ff);

    # convert the frames: mplayer writes them out as XXXXXXXX.png starting
    # with frame 1, and we want frameXXXXXX.ppm instead starting at 0:
    # CAUTION: will not work with grayscale of B/W movies!
    print STDERR "Converting frames ${range}to PPM...\n";
    $ii = $fbeg + 1; $oo = 0;
    $fr = sprintf("$tdir/$tdir2/%08d.png", $ii);
    while(-s $fr && $oo <= $fend) {
        $newfr = sprintf("$tdir/$tdir2/frame%06d.ppm", $oo);
        system("$cmd_conv < $fr > $newfr"); unlink($fr); $ii ++;
        $fr = sprintf("$tdir/$tdir2/%08d.png", $ii); $oo ++;
    }
    $oo --; print STDERR "$oo frames converted.\n";
    system("/bin/rm -f $tdir/$tdir2/*.png");

    # move result back to current directory:
    if ($dir) { $mname = "$dir/$base.mpg"; }
    else { $mname = "$base.mpg"; }

    # create mpeg-1
    # get output frame range:
    $off = `/bin/ls $tdir/$tdir2 | grep 'frame......\\.ppm'|head -1`;
    $off = substr($off,5,6);
    $olf = `/bin/ls $tdir/$tdir2 | grep 'frame......\\.ppm'|tail -1`;
    $olf = substr($olf,5,6);
    print STDERR "### Output frames: $off - $olf\n";
    # get input file size:
    open F, "$tdir/$tdir2/frame$off.ppm" ||
        die "Cannot read frame$off.ppm\n";
    $go = 2; $siz = "";
    while($go) {
        $x = <F>; chomp $x;
        if (substr($x, 0, 1) ne '#')
        { $go --; if ($go == 0) { $x =~ s/\s+/x/g; $siz = $x} }
    }
    close F; if ($siz eq "") {die "Bogus file format for frame$off.ppm\n";}
    print STDERR "### Frame size: $siz\n";

    if ($do_mpghq) {
        encode_moviehq($mname, $off, $olf, $siz, "$tdir/$tdir2", 'frame');
    } else {
        encode_movie($mname, $off, $olf, $siz, "$tdir/$tdir2", 'frame');
    }

    # delete the frames:
    print STDERR "Cleaning up...\n";
    system("/bin/rm -rf $tdir/$tdir2");
}

# delete temp directory:
system("/bin/rm -rf $tdir");

######################################################################
sub encode_movie {  # name, first_frame, last_frame, size, tdir, fram
    my $pname; $pname = "$_[4]/param.$$";
    open FF, ">$pname" || die "Cannot write $pname\n";
    print FF <<EOF;
PATTERN          IBBPBBPBBPBBPBB
OUTPUT           $_[0]
SIZE             $_[3]
INPUT_DIR        $_[4]
BASE_FILE_FORMAT PPM
GOP_SIZE         30
SLICES_PER_FRAME 1
PIXEL                 HALF
RANGE                 10
PSEARCH_ALG         LOGARITHMIC
BSEARCH_ALG         CROSS2
IQSCALE                 8
PQSCALE                 10
BQSCALE                 25
FORCE_ENCODE_LAST_FRAME 1
REFERENCE_FRAME         ORIGINAL
INPUT_CONVERT         \*
INPUT
$_[5]\*.ppm        [$_[1]-$_[2]]
END_INPUT
EOF

    close FF; system('sync');
    print STDERR "### Encoding mpeg movie into $_[0]\n";
    system("mpeg_encode $pname");
    unlink($pname);
}

######################################################################
sub encode_moviehq {  # name, first_frame, last_frame, size, tdir, fram
    my $pname; $pname = "$_[4]/param.$$";
    open FF, ">$pname" || die "Cannot write $pname\n";
    print FF <<EOF;
PATTERN          IBBPBBPBBPBBPBB
OUTPUT           $_[0]
SIZE             $_[3]
INPUT_DIR        $_[4]
BASE_FILE_FORMAT PPM
GOP_SIZE         30
SLICES_PER_FRAME 1
PIXEL                 HALF
RANGE                 10
PSEARCH_ALG         LOGARITHMIC
BSEARCH_ALG         CROSS2
IQSCALE                 1
PQSCALE                 1
BQSCALE                 1
FORCE_ENCODE_LAST_FRAME 1
REFERENCE_FRAME         ORIGINAL
INPUT_CONVERT         \*
INPUT
$_[5]\*.ppm        [$_[1]-$_[2]]
END_INPUT
EOF

    close FF; system('sync');
    print STDERR "### Encoding mpeg movie into $_[0]\n";
    system("mpeg_encode $pname");
    unlink($pname);
}
