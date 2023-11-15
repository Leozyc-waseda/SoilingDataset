#!/usr/bin/perl -w

# $Id: run_test_suite.pl 10003 2008-07-30 01:58:54Z icore $

use strict;
use testrun;

my $status;

# if very first option is --long, run testlong_*.pl scripts instead of
# default test_*.pl scripts. If it is --all run them all:
my $glob = "./test_*.pl";
if (scalar(@ARGV) > 0) {
    if    ($ARGV[0] eq '--long')  { $glob = "./testlong_*.pl"; shift; }
    elsif ($ARGV[0] eq '--short') { $glob = "./test_*.pl"; shift; }
    elsif ($ARGV[0] eq '--all')   { $glob = "./testlong_*.pl ./test_*.pl"; shift; }
}

# if next option is --threads=XXX, use parallel execution:
if (scalar(@ARGV) > 0 && $ARGV[0] =~ m/^--threads=/) {
    # multi-threaded execution:
    my @x = split(/=/, shift);

    # do we want to use a benchmark logfile?
    if (scalar(@ARGV) > 0 && $ARGV[0] =~ m/^--bench=/) {
	my @xx = split(/=/, shift);
	$status = testrun::run_matching_scripts_parallel($glob, $x[1], $xx[1]);
    } else {
	$status = testrun::run_matching_scripts_parallel($glob, $x[1]);
    }
} else {
    # do we want to create a benchmark logfile?
    if (scalar(@ARGV) > 0 && $ARGV[0] =~ m/^--writebench=/) {
	my @x = split(/=/, shift);
	$status = testrun::benchmark_scripts($glob, $x[1]);
    } else {
	# single-threaded execution:
	$status = testrun::run_matching_scripts($glob);
    }
}
exit $status;
