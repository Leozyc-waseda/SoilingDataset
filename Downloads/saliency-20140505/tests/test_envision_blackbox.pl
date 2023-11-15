#!/usr/bin/perl -w

# $Id: test_envision_blackbox.pl 8341 2007-05-04 18:49:06Z rjpeters $

# This Perl script is a "black box" test driver. The idea is that we
# run the executable with different sets of command-line options, and
# compare the output files with reference files that are stored in the
# ref/ subdirectory. See blackbox.pm for implementation details.

use strict;

use blackbox;
use invt_config;

# here is the key to the test suite: the database of command-line
# options along with the expected output files for those option sets

# note that we're doing some implicit tests for proper parsing of
# --in/--out/--io arguments here, trying different combinations of
# things (e.g., try --in/--out vs. --io, try with and without an
# explicit 'raster:' prefix)

my @tests =
    (
     {
	 name  => 'env-basic--1',
	 args  => ['../inputs/ezframe', 'envout', '0', '1'],
	 files => ['envout-color000000.pnm',
		   'envout-color000001.pnm',
		   'envout-flicker000001.pnm',
		   'envout-intens000000.pnm',
		   'envout-intens000001.pnm',
		   'envout-motion000001.pnm',
		   'envout-ori000000.pnm',
		   'envout-ori000001.pnm',
		   'envout-vcx000000.pnm',
		   'envout-vcx000001.pnm'],
     },

     {
	 name  => 'env-multithreaded--1',
	 args  => ['../inputs/ezframe', 'envout', '0', '1', '1'],
	 sharerefs => 'env-basic--1',
	 files => ['envout-color000000.pnm',
		   'envout-color000001.pnm',
		   'envout-flicker000001.pnm',
		   'envout-intens000000.pnm',
		   'envout-intens000001.pnm',
		   'envout-motion000001.pnm',
		   'envout-ori000000.pnm',
		   'envout-ori000001.pnm',
		   'envout-vcx000000.pnm',
		   'envout-vcx000001.pnm'],
     },

     );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/envision", @tests);
