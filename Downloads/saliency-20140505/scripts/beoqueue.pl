#!/usr/bin/perl

# USAGE: beoqueue.pl [options] "<command>" "<args1>" ... "<argsN>"

# will queue <command> <args1> ... <argsP> on n01, and so on
# OPTIONS:
# -d add 1 second delay between job start on each node (multiple d for more)
# -n "node1 node2 node3 ..." use given list of nodes instead of /etc/nodes
#   each node can be specified as either just a node name (will run one process
#   on that node, note that the same name can appear several times in the list
#   for nodes with multiple CPU cores), or node:n to run n processes on that node,
#   or node: to run as many processes on that node as it has CPU cores according
#   to /proc/cpuinfo on that node.
# -q to run the children quietly (no aterm)
# -f <argfile> to read args from a file (one per line)

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
## $Id: beoqueue.pl 13004 2010-03-11 19:32:28Z itti $
##
use POSIX;
$SIG{CHLD} = \&reaper; $SIG{USR1} = \&error; $SIG{USR2} = \&warning;
use strict;

my $ter    = "aterm -fn 5x7 -fg black -bg white";    # terminal to use
my $muname = "/tmp/beoqueue.mutex.$$";     # mutex (may be unnecessary)
my $procid = POSIX::getpid();

# These settings are for my 30in display
my $txoff = 5 + 150;  # X offset for the terminals
my $twidth = 435;     # terminal width
my $tyoff = 22;       # Y offset for the terminals
my $theight = 202;    # terminal width
my $tnx = 5;          # number of terminals across

##############################################################################
my $delay = 0; my $nod = ""; my $quiet = 0; my $argfname = "";
while(substr($ARGV[0], 0, 1) eq '-') {
    my $ar = shift;
    if ($ar =~ /(d+)/) { $delay += length($ar) - 1; }
    if ($ar =~ /n/) { $nod = shift; }
    if ($ar =~ /q/) { $quiet = 1; }
    if ($ar =~ /f/) { $argfname = shift; }
}

# get the command and jobs:
my $cmd = shift; my @jlist;
if ($argfname) { @jlist = `/bin/cat $argfname`; } else { @jlist = @ARGV; }
my $njob = $#jlist + 1; my $job = 0;

# get the nodes:
msg("Calculating number of nodes...");
my @nods;
if ($nod eq "") {
    if (-s "$ENV{HOME}/.beonodes") { @nods = `/bin/cat $ENV{HOME}/.beonodes`; }
    elsif (-s '/etc/beonodes') { @nods = `/bin/cat /etc/beonodes`; }
    else { die 'Cannot find list of nodes -- ABORT'; }
} else { @nods = split(/\s+/, $nod); }

# parse the node list and expand the node:n and node: cases:
my @nodes;
foreach my $nnn (@nods) {
    chomp($nnn);
    next if substr($nnn, 0, 1) eq '#'; # skip comments
    if ($nnn =~ m/:/) {
	# node:n or node: specification:
	my @tmp = split(/:/, $nnn);
	if ($#tmp == 0) {
	    # node: specification, let's get the number of cores of that node:
	    my $ncpu = `ssh -x -n $tmp[0] "cat /proc/cpuinfo | grep \"^processor" | wc -l"` + 0;
	    while($ncpu) { push(@nodes, $tmp[0]); $ncpu --; }
	} elsif ($#tmp == 1) {
	    # node:n specification, use that node n times:
	    while($tmp[1]) { push(@nodes, $tmp[0]); $tmp[1] --; }
	} else { die "Invalid node specification: $nnn\n"; }
    } else {
	# just a plain node name, push it onto the list:
	push(@nodes, $nnn);
    }
}

my $nnod = $#nodes + 1;
msg("Executing $njob jobs on $nnod nodes.");
msg("Command: $cmd");

# initialize our table of child PIDs:
my $ii = 0; my @plist; while($ii <= $#nodes) { $plist[$ii++] = -1; }

# get a mutex going:
open MUTEX, ">$muname" || die "Cannot write $muname: $!\n";

# go over every argument:
my $done = 0;
while(my $ar = shift(@jlist)) {
    my $todo = 1; chomp $ar;
    while($todo) {
        # find a node that is not running anything, i.e., is not in our plist:
        my $nn = ""; my $ii = 0; my $found = 0;
        while($ii <= $#plist && $found == 0) {
            if ($plist[$ii] == -1) { $nn = $nodes[$ii]; $found = 1;}
            else { $ii ++; }
        }
        if ($nn eq "") { sleep(1); } # everybody is busy
        else {
            $job ++; # used to display job number
            msg("Dispatching $ar to $nn [$job/$njob]...");

            my $res = fork();
            if (not defined $res) { die "Cannot fork: $!\n"; }
            elsif ($res == 0) { # we are the child
                if ($quiet) { # run quietly (no aterm)
                    if ($nn eq 'localhost') {
                        system("$cmd $ar");
                    } else {
                        system("ssh -x -n $nn \"$cmd $ar\"");
                    }
                } else {      # run in an aterm
                    # figure out where to place the aterm:
                    my $x = $ii % $tnx;
                    my $y = int($ii / $tnx);

                    $x = $txoff + $x * $twidth;
                    $y = $tyoff + $y * $theight;

                    if ($nn eq 'localhost') {
                        system("$ter -geometry +$x+$y -title $nn -e $cmd $ar");
                    } else {
                        system("$ter -geometry +$x+$y -title $nn -e ssh -x -n $nn \"$cmd $ar\"");
                    }
                }
                exit 0;
            } else {
                flock MUTEX, 2;  # get protected access to plist
                $plist[$ii] = $res; # this node has been assigned
                flock MUTEX, 8;  # free plist
                $todo = 0; # just took care of that command-line arg.
            }
            if ($delay) { sleep $delay; }
        }
    }
}

# wait for all children to complete:
my $found = 1;
while($found) {
    sleep(1); $found = 0; $ii = 0;
    while($ii <= $#plist) { if ($plist[$ii++] != -1) { $found = 1; } }
}

msg("All jobs complete. Done.");

close MUTEX; unlink $muname;

######################################################################
sub reaper {   # reap dead children
    my $pid; my $n;
    while (($pid = waitpid(-1, &WNOHANG)) > 0) {
        if (WIFEXITED($?)) {
            my $jj = 0;
            while($jj <= $#plist) {
                if ($plist[$jj] == $pid) {
		    $done ++;
                    msg("Job completed on $nodes[$jj] - $done/$njob jobs done so far.");

                    # update the plist (protected by a mutex):
                    flock MUTEX, 2;  # get protected access to plist
                    $plist[$jj] = -1;
                    flock MUTEX, 8;  # free plist
                }
                $jj ++;
            }
        }
    }
    $SIG{CHLD} = \&reaper;
}

######################################################################
sub error {
    msg("FATAL ERROR recieved from child node");
    die "STOPPED!\n";
}

######################################################################
sub warning {
    msg("WARNING received from child node");
}

######################################################################
sub msg { # string
    my $dat = `/bin/date +"%y%m%d-%H:%M:%S"`; chomp $dat;
    print STDERR "BEOQUEUE $dat - $_[0]\n";
}
