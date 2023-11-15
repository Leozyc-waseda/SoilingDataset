#!/usr/bin/perl

# USAGE: beoqueue.pl [options] "<command>" "<args1>" ... "<argsN>"

# will queue <command> <args1> ... <argsP> on n01, and so on
# OPTIONS:
# -d add 1 second delay between job start on each node (multiple d for more)
# -n "node1 node2 node3 ..." use given list of nodes instead of /etc/nodes
# -q to run the children quietly (no aterm)

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
## $Id: beoqueue.pl 7042 2006-06-27 01:51:40Z mundhenk $
##
use POSIX; 
use IO::Socket;  
$SIG{CHLD} = \&reaper; $SIG{USR1} = \&error; $SIG{USR2} = \&warning;
use strict;

my $ter    = "aterm -fn 5x7";              # terminal to use
my $muname = "/tmp/beoqueue.mutex.$$";     # mutex (may be unnecessary)
my $procid = POSIX::getpid();
my $maxNodeErrors = 10;


##############################################################################
my $delay = 0; my $nod = ""; my $quiet = 0; my $shell = 0; my $perl_soc = 0;
print("Parsing command line...\n");
while(substr($ARGV[0], 0, 1) eq '-') {
    my $ar = shift;
    if ($ar =~ /(d+)/) { $delay += length($ar) - 1; }
    if ($ar =~ /n/)    { $nod           = shift; }
    if ($ar =~ /m/)    { $maxNodeErrors = shift; }
    if ($ar =~ /q/)    { $quiet         = 1; }
    if ($ar =~ /s/)    { $shell         = 1; }
    if ($ar =~ /p/)    { $perl_soc      = 1; }
}

$quiet = 1;
# get the command and number of jobs:
my $cmd = shift; my $njob = $#ARGV + 1; my $job = 0;

# get the nodes:
if ($nod eq "") { $nod = `cat /etc/nodes`|| die "FATAL>>> Cannot find /etc/nodes: $!\n";}
my @nodes = split(/\s+/, $nod); my $nnod = $#nodes + 1;
msg("Executing $njob jobs on $nnod nodes.");
msg("Command: $cmd");

# initialize our table of child PIDs:
my $ii = 0; my @plist; my @unfinished; my @openjob; my %nodeErrCount;
while($ii <= $#nodes) { $plist[$ii++] = -1; }
# initialize hash table for node error counting
$ii = 0;
while($ii <= $#nodes) { $nodeErrCount{$nodes[$ii++]} = 0; }

# get a mutex going:
open MUTEX, ">$muname" || die "FATAL>>> Cannot write mutex file $muname: $!\n";

# go over every argument:
while(my $ar = shift) {
    push(@unfinished, $ar);              # create a local list so we can repeat failed jobs
    while(my $ujob = pop(@unfinished)) { # if no jobs fail, this has only one job in it
	my $todo = 1;
	while($todo) {
	    # find a node that is not running anything, i.e., is not in our plist:
	    my $nn = ""; my $ii = 0; my $found = 0;
	    
	    while($ii <= $#plist && $found == 0) {
		if (($plist[$ii] == -1) && ($nodeErrCount{$nodes[$ii]} < $maxNodeErrors))
		    { $nn = $nodes[$ii]; $found = 1;}
		else { $ii ++; }
	    }
	    if ($nn eq "") { sleep(1); } # everybody is busy
	    else {
		$job ++; # used to display job number
		msg("Dispatching $ujob to $nn [$job/$njob]...");
	    
		my $res = fork();
		if (not defined $res) { die "FATAL>>> Cannot fork: $!\n"; }
		elsif ($res == 0) {    # we are the child
		    my @cmdres;        # the result of the command
		    if ($shell) {      # run locally in perl
			print("> perl $cmd $ujob \n");
			@cmdres    = qx"perl $cmd $ujob 2>&1";
			$cmdres[1] = "perl $cmd $ujob 2>&1";
			$cmdres[2] = "perl";
		    } elsif ($perl_soc) { # use perl sockets to talk to a remote perl host
			print("> perl_soc $nn $ii $job $cmd $ujob \n");
			@cmdres    = call_perl_soc($nn,$ii,$job,$cmd,$ujob);
			$cmdres[1] = "perl_soc $nn $ii $job $cmd $ujob 2>&1";
			$cmdres[2] = "soc";		     
		    } elsif ($quiet) { # run quietly (no aterm)
			print("> rsh -n $nn \"$cmd $ujob\" \n");
			@cmdres    = qx"rsh -n $nn \"$cmd $ujob\" 2>&1";
			$cmdres[1] = "rsh -n $nn \"$cmd $ujob\" 2>&1";
			$cmdres[2] = "rsh";
		    } else {           # run in an aterm
			print("> rsh -n $nn \"$ter -title $nn -e $cmd $ar\"");
			@cmdres    = qx"rsh -n $nn \"$ter -title $nn -e $cmd $ujob\" 2>&1";
			$cmdres[1] = "rsh -n $nn \"$ter -title $nn -e $cmd $ujob\" 2>&1";
			$cmdres[2] = "rsh";
		    }
		    if(checkCommand(@cmdres)) { # error returned, exit status is set to 1 for this child
			print("WARNING>>> Error in shell exit value\n"); 
			exit 1;
		    }
		    else {
			exit 0;
		    }
		} else {
		    flock MUTEX, 2;          # get protected access to plist
		    $plist[$ii]   = $res;    # this node has been assigned
		    $openjob[$ii] = $ujob;   # this job is running
		    flock MUTEX, 8;          # free plist and errList
		    $todo = 0;               # just took care of that command-line arg.	       
		}
		if ($delay) { sleep $delay; }
	    }
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
# This function is called EVERY TIME a child exits. It frees up the 
# node for future use. It will check the exit state of the child. If
# the exit is not 0, then it will set to retry the job 
sub reaper {   # reap dead children
    my $pid; my $n;
    while (($pid = waitpid(-1, &WNOHANG)) > 0) { # get the exiting child PID
	if (WIFEXITED($?)) { 
	    if(WSTOPSIG($?) == 0) {              # everything went OK, just reclaim and count us all done
		my $jj = 0;
		while($jj <= $#plist) {
		    if ($plist[$jj] == $pid) {
			msg("Job completed on $nodes[$jj]");
			# update the plist (protected by a mutex):
			flock MUTEX, 2;  # get protected access to plist
			$plist[$jj] = -1;
			flock MUTEX, 8;  # free plist
		    }
		    $jj ++;
		}
	    }
	    else { #error in this job (exit was 1)
		my $jj = 0;
		while($jj <= $#plist) {
		    if ($plist[$jj] == $pid) {
			msg("Job NOT completed on $nodes[$jj] - Job requeued");
			
			# update the plist (protected by a mutex):
			flock MUTEX, 2;                    # get protected access to plist
			$plist[$jj] = -1;
			$nodeErrCount{$nodes[$jj]}++;      # add this nodes error
			if($nodeErrCount{$nodes[$jj]} >= $maxNodeErrors)
			{
			    print("NOTICE>>> MAX ERRORS >= $maxNodeErrors ON NODE $nodes[$jj]\n");
			    print("NOTICE>>> NO FURTHER JOBS WILL BE SENT TO NODE\n");
			}
			push(@unfinished, $openjob[$jj]);  # relist this job to be done
			$job--;
			flock MUTEX, 8;                    # free plist
		    }
		    $jj ++;
		}
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
    chomp();
    my $dat = `/bin/date +"%y%m%d-%H:%M:%S"`; chomp($dat);
    print STDERR "BEOQUEUE $dat - $_[0]\n";
}

######################################################################
sub checkCommand {
    my $result = 0;

    if($_[2] eq "rsh")
    {
	if($_[0] =~ "Permission")
	{
	    chomp($_[0]);
	    print("WARNING>>> Shell Command Failed:\n");
	    print("WARNING>>> $_[1]\n"); 
	    print("WARNING>>> Error 01: Permission Denied\n");
	    print("WARNING>>> \"$_[0]\" \n");
	    print("WARNING>>> Will Retry Job\n");
	    $result = 1;
	}
	elsif($_[0] =~ "rcmd")
	{
	    chomp($_[0]);
	    print("WARNING>>> Shell Command Failed:\n"); 
	    print("WARNING>>> $_[1]\n"); 
	    print("WARNING>>> Error 02: rcmd failure\n");
	    print("WARNING>>> \"$_[0]\" \n");
	    print("WARNING>>> Will Retry Job\n");
	    $result = 1;
	}  
	elsif($_[0] =~ "circuit")
	{
	    chomp($_[0]);
	    print("WARNING>>> Shell Command Failed:\n"); 
	    print("WARNING>>> $_[1]\n"); 
	    print("WARNING>>> Error 03: Circuit failure\n");
	    print("WARNING>>> \"$_[0]\" \n");
	    print("WARNING>>> Will Retry Job\n");
	    $result = 1;
	}
	
	elsif(($_[0] =~ "route") || ($_[0] =~ "host"))
	{
	    chomp($_[0]);
	    print("WARNING>>> Shell Command Failed:\n"); 
	    print("WARNING>>> $_[1]\n"); 
	    print("WARNING>>> Error 04: Host Route failure\n");
	    print("WARNING>>> \"$_[0]\" \n");
	    print("WARNING>>> Will Retry Job\n");
	    $result = 1;
	} 
	elsif($_[0] =~ m/\S{4,}/) # match 4 or more non-whitespace chars
	{
	    chomp($_[0]);
	    print("WARNING>>> Shell Command Failed:\n"); 
	    print("WARNING>>> $_[1]\n"); 
	    print("WARNING>>> Error 05: Unspecified failure\n");
	    print("WARNING>>> \"$_[0]\" \n");
	    print("WARNING>>> Will Retry Job\n");
	    $result = 1;
	} 
    }
    elsif($_[2] eq "soc")
    {
	if($_[0] =~ "unknown_failure")
	{
	    chomp($_[0]);
	    print("WARNING>>> PERL Socket Command Failed:\n"); 
	    print("WARNING>>> $_[1]\n"); 
	    print("WARNING>>> Error 06: Unknown failure\n");
	    print("WARNING>>> \"$_[0]\" \n");
	    print("WARNING>>> Will Retry Job\n");
	    $result = 1;
	}
    }	
    return $result;
}

######################################################################
# run the script as a perl socket
# @cmdres    = call_perl_soc($nn,$nodejob{$nodes[$ii]},$job,$cmd,$ujob);
sub call_perl_soc
{
    my $node    = $_[0];
    my $nodejob = $_[1];
    my $job     = $_[2];
    my $cmd     = $_[3];
    my $ujob    = $_[4];

    my $BASE_OUT = 50000 + $nodejob;
    my $BASE_IN  = 50500 + $nodejob;


    # Open outgoing socket
    my $sock = new IO::Socket::INET ( 
	PeerAddr => $node, 
	PeerPort => $BASE_OUT, 
	Proto => 'tcp',
	); 
    die "Could not create socket: $!\n" unless $sock;

    # open incoming socket
    my $isock = new IO::Socket::INET ( 
	LocalHost => '', 
	LocalPort => $BASE_IN, 
	Proto => 'tcp', 
	Listen => 1, 
	Reuse => 1, 
	);
    die "Could not create socket: $!\n" unless $isock;

    # get info about me
    my @hname      = qx'echo $HOSTNAME';
    my @shn        = split(/\n/,$hname[0]);
    my $HOST       = $shn[0];
    my $packed_ip  = gethostbyname($HOST);
    my $IP;
    if (defined $packed_ip) { $IP = inet_ntoa($packed_ip); }

    # set up the job to run
    my $ID     = "$job";
    my $MTYPE  = "COMMAND";
    my $MCONT  = "perl $cmd $ujob 2>&1";
    my $RET    = 1; 

    # send job to node
    print $sock "$HOST,$IP,$ID,$MTYPE,$MCONT,$RET";
    close($sock);

    my @CMDRES = {"unknown_failure"};
    my $i      = 0;

    my $new_sock = $isock->accept(); 
    while(<$new_sock>) 
    { 
	$CMDRES[$i] = $_;
	$i++;
    } 
    close($isock);

    return @CMDRES;
}
