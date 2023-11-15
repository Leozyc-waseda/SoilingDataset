#!/usr/bin/perl

# USAGE: eyetrace.pl [OPTS] <video.mpg> <eyetrace.eye> <output>
#
# Will paint the eyetrace onto the video. If the eyetrace is in .eyeS
# format, saccade markers will be painted as well.
#
# [OPTS] can be any valid long options that streamision accepts (should
# start with '--'). An additional option is provided:
# --framedur=x to select a video frame duration of x ms (default: 33.185)
#
# the final output name will be <output>T.mpg
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


$fdur = 33.185; # default frame duration
$trash = 240;   # default initial records to trash

# parse the command line:
$in = ''; $eye = ''; $out = ''; $opts = '';
foreach $a (@ARGV)
{
    if (substr($a, 0, 2) eq '--')
    {
        ($o, $v) = split(/=/, $a, 2);
        if ($o eq '--framedur')
        { $fdur = $v; print STDERR "Using frame duration of ${v}ms\n"; }
        else { $opts .= "$a "; }
    }
    elsif ($in eq '') { $in = $a; }
    elsif ($eye eq '') { $eye = $a; }
    elsif ($out eq '') { $out = $a; }
    else { die "Unrecognized argument: $a -- ABORT\n"; }
}

# let's process it:
system("ezvision --nouse-fpe --nouse-random --logverb=Error ".
       "--in=$in --input-frames=\@${fdur}ms ".
       "--out=mpeg:$out --out=display --output-frames=\@${fdur}ms ".
       "--vc-type=None --sm-type=None --trm-type=None ".
       "--wta-type=None --shape-estim-mode=None ".
       "--sv-type=EyeMvt ".
       "--svem-display-sacnum ".
       "--save-trajectory ".
       "--fovea-radius=64 ".    # about 5.6deg diameter @ 23ppd
       "--maxnorm-type=FancyOne ".
       "--maxcache-size=0 ".  # delay cache: 0 for 0ms, 12 for 50ms
       "--eye-trash=$trash ".
       "--eye-fname=$eye $opts ");
