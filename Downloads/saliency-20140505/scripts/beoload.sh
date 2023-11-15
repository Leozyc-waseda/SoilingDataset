#!/bin/sh

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
## $Id: beoload.sh 13908 2010-09-10 02:06:28Z itti $
##

# find a list of nodes:
if [ -s "${HOME}/.beonodes" ]; then nodes="${HOME}/.beonodes"
elif [ -s '/etc/beonodes' ]; then nodes='/etc/beonodes'
else echo 'Cannot find list of nodes -- ABORT'; exit 1; fi

# don't try to write scripts like this at home, but this seems to work!
cmd="/bin/bash -c 'lo=\`cat /proc/loadavg | awk \"{ print \\\\\$1 \\\" \\\" \
\\\\\$2 \\\" \\\" \\\\\$3 \\\" \\\" \\\\\$4 }\"\`; dsk=\`df -h /home | grep \
home | awk \"{ print \\\\\$4}\"\`; if [ \${lo:0:3} != 0.0 ]; then \
echo -en \"\\033[1;31m\"; fi; echo -e \" \$lo\\033[0;39m   \$dsk\tfree  \" \
\`date\`   \`uname -r\`'"

for n in `/bin/grep -v "#" $nodes`; do
    echo -n "$n "
    ssh -x $n "$cmd"
done
