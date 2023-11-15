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
	 name  => "combine-optigains--1",
	 args  => ['../ref/ez-optigains--1--berry1.stsd',
		   '../ref/ez-optigains--2--berry2.stsd',
		   '--save-to=berry12.pmap'],
	 files => ['berry12.pmap'],
     },
     {
	 name  => "combine-optigains--2",
	 args  => ['../ref/ez-optigains--1--berry1.stsd',
		   '--save-to=berry1.pmap'],
	 files => ['berry1.pmap'],
     },
     );

# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"
blackbox::run("$invt_config::exec_prefix/bin/app-combineOptimalGains", @tests);
