#!/usr/bin/perl -w

# $Id: test_ImageEqual_blackbox.pl 8072 2007-01-18 20:00:51Z rjpeters $

# This Perl script is a "black box" test driver. The idea is that we
# run the executable with different sets of command-line options, and
# compare the output files with reference files that are stored in the
# ref/ subdirectory. See blackbox.pm for implementation details.

use strict;

use blackbox;
use invt_config;

my @tests =
    (
     {
	 name  => "sift-simple--1",
	 args  => ['Simple',
		   '../inputs/berry1.png',
		   '../inputs/berry2.png',
		   'matches.pnm',
		   'fused.pnm',
		   '>status.txt 2>&1'],
	 files => ['matches.pnm', 'fused.pnm', 'status.txt'],
     },
     {
	 name  => "sift-kdtree--1",
	 args  => ['KDTree',
		   '../inputs/berry3.png',
		   '../inputs/berry1.png',
		   'matches.pnm',
		   'fused.pnm',
		   '>status.txt 2>&1'],
	 files => ['matches.pnm', 'fused.pnm', 'status.txt'],
     },
     {
	 name  => "sift-kdbbf--1",
	 args  => ['KDBBF',
		   '../inputs/berry3.png',
		   '../inputs/berry2.png',
		   'matches.pnm',
		   'fused.pnm',
		   '>status.txt 2>&1'],
	 files => ['matches.pnm', 'fused.pnm', 'status.txt'],
     },
     );

# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/test-SIFTimageMatch", @tests);
