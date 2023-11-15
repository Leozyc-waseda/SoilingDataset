#!/usr/bin/perl -w

# $Id: test_ImageEqual_blackbox.pl 7805 2007-01-26 19:41:57Z rjpeters $

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
	 name  => "ieq-gray8--1",
	 args  => ['../inputs/gray-8-bit-001.pgm',
		   '../inputs/gray-8-bit-001.png',
		   'ieq-status1.txt'],
	 files => ['ieq-status1.txt'],
     },

     {
	 name  => "ieq-rgb24--1",
	 args  => ['../inputs/rgb-24-bit-001.ppm',
		   '../inputs/rgb-24-bit-001.png',
		   'ieq-status2.txt'],
	 files => ['ieq-status2.txt'],
     },

     {
	 name  => "ieq-rgb-palette--1",
	 args  => ['../inputs/rgb-24-bit-001.ppm',
		   '../inputs/palette-8-bit-001.png',
		   'ieq-status3.txt'],
	 files => ['ieq-status3.txt'],
     },

     {
	 name  => "ieq-rgb-gz--1",
	 args  => ['../inputs/rgb-24-bit-001.ppm',
		   '../inputs/rgb-24-bit-001.pnm.gz',
		   'ieq-status4.txt'],
	 files => ['ieq-status4.txt'],
     },

     {
	 name  => "ieq-rgb-bz2--1",
	 args  => ['../inputs/rgb-24-bit-001.ppm',
		   '../inputs/rgb-24-bit-001.pnm.bz2',
		   'ieq-status5.txt'],
	 files => ['ieq-status5.txt'],
     },

     {
	 name  => "ieq-bw1--1",
	 args  => ['../inputs/bw-1-bit-raw.pbm',
		   '../inputs/bw-1-bit-raw.pgm',
		   'ieq-status6.txt'],
	 files => ['ieq-status6.txt'],
     },

     {
	 name  => "ieq-bw1--2",
	 args  => ['../inputs/bw-1-bit-ascii.pbm',
		   '../inputs/bw-1-bit-raw.pgm',
		   'ieq-status7.txt'],
	 files => ['ieq-status7.txt'],
     },
     );

# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/test-ImageEqual", @tests);
