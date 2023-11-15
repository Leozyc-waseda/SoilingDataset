#!/usr/bin/perl -w

# $Id: test_retina_blackbox.pl 6323 2006-02-23 19:38:44Z rjpeters $

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
	 name  => "noopt--1",
	 args  => [qw(testpic001.pnm retinaout.ppm)],
	 files => ["retinaout.ppm"],
     },

     {
	 name  => "blind--1",
	 args  => [qw(-b testpic001.pnm retinaout.ppm)],
	 files => ["retinaout.ppm"],
     },

     {
	 name  => "bluecones--1",
	 args  => [qw(-f testpic001.pnm retinaout.ppm)],
	 files => ["retinaout.ppm"],
     },

     {
	 name  => "allopt--1",
	 args  => [qw(-bf testpic001.pnm retinaout.ppm 100 100)],
	 files => ["retinaout.ppm"],
     },
     );

# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/retina", @tests);
