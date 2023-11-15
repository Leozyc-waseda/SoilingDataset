# A Perl module for queueing a series of jobs to be dispatched to run
# on one or more local and/or remote nodes

# example usage:

# my $cmd = "/some/cmd";
# my @args = ("a b c", "1 2 3", "foo bar baz");
# my @nodes = ("localhost", "remote1");
# jobqueue::queue({
#     delay => 1,           # number of seconds to pause after starting each job before starting the next one
#     quiet => 1,           # if 0, then run each job in a separate terminal window
#     command => $cmd,  # executable program for each job
#     jobs => \@args,   # list of command-line args, one for each job
#     nodes => \@nodes, # list of nodes to which jobs are dispatched
# });

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
## $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/scripts/jobqueue.pm $
## $Id: jobqueue.pm 10337 2008-10-13 17:24:09Z itti $
##

package jobqueue;

require Exporter;

@ISA = qw(Exporter);

use strict;
use POSIX;
$SIG{USR1} = \&error;
$SIG{USR2} = \&warning;

my @plist;

sub queue {

    my ($args) = @_;

    my $delay = $args->{delay};
    my $quiet = $args->{quiet};
    my $cmd = $args->{command};

    my $jobs = $args->{jobs};
    my $njob = scalar(@$jobs);

    my $nodes = $args->{nodes};
    my $nnod = scalar(@$nodes);

    my $remoteshell = "ssh -n";
    if (defined $args->{remoteshell})
    {
        $remoteshell = $args->{remoteshell};
    }

    my $ter    = "aterm -fn 5x7 -fg black -bg white";    # terminal to use
    my $procid = POSIX::getpid();

    # These settings are for my 30in display
    my $txoff = 5 + 150;     # X offset for the terminals
    my $twidth = 435;  # terminal width
    my $tyoff = 22;    # Y offset for the terminals
    my $theight = 202; # terminal width
    my $tnx = 5;       # number of terminals across

    my @termposx;
    my @termposy;
    my @freenodes;

    for (my $ii = 0; $ii < scalar(@$nodes); $ii++)
    {
        # figure out where to place the aterm windows:

        my $x = $ii % $tnx;
        my $y = int($ii / $tnx);

        $termposx[$ii] = $txoff + $x * $twidth;
        $termposy[$ii] = $tyoff + $y * $theight;

        push @freenodes, $ii;
    }

    my $job = 0;

    msg("Executing $njob jobs on $nnod nodes.");
    msg("Command: $cmd");
    msg("Remote shell: $remoteshell");

    my %childpids;

    # go over every argument:
    foreach my $ar (@$jobs)
    {
        if (scalar(@freenodes) == 0)
        {
            # everybody is busy; let's wait for one of our child
            # processes to exit:

            my $pid = wait();
            my $freenode = $childpids{$pid};
            push @freenodes, $freenode;
            delete $childpids{$pid};
            msg("Job completed on node #$freenode ($nodes->[$freenode])");
        }

        my $nodeid = shift(@freenodes);
        my $nn = $nodes->[$nodeid];

        $job++; # used to display job number
        msg("Dispatching job $job/$njob to $nn [$cmd $ar]...");

        my $res = fork();
        if (not defined $res)
        {
            die "Cannot fork: $!\n";
        }
        elsif ($res == 0)  # we are the child
        {
            if ($quiet)  # run quietly (no aterm)
            {
                if ($nn eq 'localhost')
                {
                    system("$cmd $ar");
                }
                else
                {
                    system("$remoteshell $nn \"$cmd $ar\"");
                }
            }
            else      # run in an aterm
            {
                # figure out where to place the aterm:
                my $x = $termposx[$nodeid];
                my $y = $termposy[$nodeid];

                $x = $txoff + $x * $twidth;
                $y = $tyoff + $y * $theight;

                if ($nn eq 'localhost')
                {
                    system("$ter -geometry +$x+$y -title $nn ".
                           "-e $cmd $ar");
                }
                else
                {
                    system("$ter -geometry +$x+$y -title $nn ".
                           "-e $remoteshell $nn \"$cmd $ar\"");
                }
            }
            exit 0;
        }
        else
        {
            $childpids{$res} = $nodeid;
        }

        if ($delay > 0) { sleep $delay; }
    }

    # wait for all children to complete:
    while ((my $pid = wait()) > 0)
    {
        my $freenode = $childpids{$pid};
        msg("Job completed on node #$freenode ($nodes->[$freenode])");
    }

    msg("All jobs complete. Done.");
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

1;
