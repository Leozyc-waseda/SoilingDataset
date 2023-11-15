#!/usr/bin/perl
#USAGE: process.pl <image.ppm> ... <image.ppm>

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
## $Id: sbprocess.pl 10337 2008-10-13 17:24:09Z itti $
##

$img = 0;
while($img <= $#ARGV) {
    $fn = $ARGV[$img]; $count = 0; ($base, $ext) = split(/\./, $fn, 2);
    if ($ext ne "ppm") {
        print "Converting $fn to $base.ppm\n";
        system("convert $fn pnm:$base.ppm");
        $fn = "$base.ppm";
    }
    open F, $fn || die "Cannot read $fn";
    while($line = <F>) {
        chomp $line;
        if (substr($line, 0, 1) ne "#") {
            if ($count == 0) {
                if ($line ne "P6") { die "$fn is Not a PPM file!"; }
            } elsif ($count == 1) {
                ($x, $y) = split(/\s+/, $line);
                print "Image size [$base]: $x x $y\n";
                goto done;
            } else { die "bug!"; }
            $count ++;
        }
    }
done:
    close F;

    $x *= 2;  # juxtapose in X
    while($x > 800) { $x /= 2; $y /= 2; }
    $x = int($x); $y = int($y);
    print "Final movie size: ${x}x${y}\n";

    system("vision -TXy --resize ${x}x${y} -R 5 --frames 1-300 ".
           "--display-map-factor 40000 $fn");
    system("rescale ../copyright.ppm $x $y copyright.ppm");
    system("fade_in copyright.ppm 301 310");
    system("mpeg_encode ../movie.param");
    system("/bin/mv movie.mpg MOVIE-$base.mpg");
    system("/bin/rm T??????.ppm");
    system("mpeg_play -dither color MOVIE-$base.mpg &");

    # ready for next image:
    $img ++;
}
system("/bin/rm copyright.ppm result_*_.txt");
