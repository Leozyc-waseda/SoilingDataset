#!/usr/bin/perl -w

# $Id: test_retinex_blackbox.pl 8019 2007-03-01 19:33:05Z rjpeters $

# This Perl script is a "black box" test driver. The idea is that we
# run the executable with different sets of command-line options, and
# compare the output files with reference files that are stored in the
# ref/ subdirectory. See blackbox.pm for implementation details.

use strict;

use blackbox;
use invt_config;

# here is the key to the test suite: the database of command-line
# options along with the expected output files for those option sets

my @tests =
    (
     {
	 name  => "retinex--1",
	 args  => ['../inputs/rgb-48-bit.pnm', '0', '1'],
	 files => ['blev0.pfm', 'glev0.pfm', 'rlev0.pfm',
		   'blev1.pfm', 'glev1.pfm', 'rlev1.pfm',
		   'blev2.pfm', 'glev2.pfm', 'rlev2.pfm',
		   'blev3.pfm', 'glev3.pfm', 'rlev3.pfm',
		   'blev4.pfm', 'glev4.pfm', 'rlev4.pfm',
		   'blev5.pfm', 'glev5.pfm', 'rlev5.pfm',
		   'blev6.pfm', 'glev6.pfm', 'rlev6.pfm',
		   'blev7.pfm', 'glev7.pfm', 'rlev7.pfm',
		   'blev8.pfm', 'glev8.pfm', 'rlev8.pfm'],
     },

     );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/test-retinex", @tests);
