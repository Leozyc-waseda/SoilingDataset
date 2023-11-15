#!/usr/bin/perl -w

# $Id: test_fzoom_blackbox.pl 8718 2007-08-24 16:20:44Z rjpeters $

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
	 name  => "fzoom-box-1",
	 args  => ['-x', '48', '-y', '90', '-f', 'b',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-triangle-1",
	 args  => ['-x', '47', '-y', '91', '-f', 't',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-bell-1",
	 args  => ['-x', '46', '-y', '92', '-f', 'q',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-bspline-1",
	 args  => ['-x', '45', '-y', '93', '-f', 'B',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-hermite-1",
	 args  => ['-x', '44', '-y', '94', '-f', 'h',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-lanczos3-1",
	 args  => ['-x', '43', '-y', '95', '-f', 'l',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-mitchell-1",
	 args  => ['-x', '42', '-y', '96', '-f', 'm',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-box--2",
	 args  => ['-x', '30', '-y', '14', '-f', 'b',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-triangle--2",
	 args  => ['-x', '29', '-y', '15', '-f', 't',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-bell--2",
	 args  => ['-x', '28', '-y', '16', '-f', 'q',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-bspline--2",
	 args  => ['-x', '27', '-y', '17', '-f', 'B',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-hermite--2",
	 args  => ['-x', '26', '-y', '18', '-f', 'h',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-lanczos3--2",
	 args  => ['-x', '25', '-y', '19', '-f', 'l',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     {
	 name  => "fzoom-mitchell--2",
	 args  => ['-x', '24', '-y', '20', '-f', 'm',
		   '../inputs/gray-8-bit-001.pgm', 'out.pgm'],
	 files => ['out.pgm'],
     },

     );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/fzoom", @tests);
