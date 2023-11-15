#!/usr/bin/perl

# USAGE: killallmyprocs.pl <killopts>
# Will kill all your processes on the current machine.
# Typically, to be used with brsh to kill hanging processes on our Beowulf:
# brsh "killallmyprocs.pl -9"   will sweep the Beowulf clean of your junk.

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
## $Id: killallmyprocs.pl 10311 2008-10-01 22:14:04Z itti $
##

$opts = ""; while($x = shift) { $opts .= "$x "; }
$kp = ""; $kn = ""; $parent = "";

# get process list:
@ps=`ps --User $ENV{'USER'} --no-headers --format pid,ppid,comm`;

# figure out pid of my parent
foreach $p (@ps) {
    chomp $p; $p =~ s/^\s+//; ($pid, $ppid, $cmd) = split(/\s+/, $p, 3);
    if ($pid == $$) { $parent = $ppid; }
}
# kill everybody except ourselves, our parent, and our children:
foreach $p (@ps) {
    chomp $p; $p =~ s/^\s+//; ($pid, $ppid, $cmd) = split(/\s+/, $p, 3);
    if ($pid != $$ && $pid != $parent && $ppid != $$)
    { $kn .= "${cmd}[$pid] "; $kp .= "$pid "; }
}

if ($kp eq "") {
    print "No process to kill...\n";
} else {
    print "Killing $kn...\n";
    system("kill $opts $kp");
}
