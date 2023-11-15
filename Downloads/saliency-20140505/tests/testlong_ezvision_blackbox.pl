#!/usr/bin/perl -w

# $Id: test_ezvision_blackbox.pl 8259 2007-02-08 21:07:00Z rjpeters $

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
	 name  => "ezlong-movietraj--1",
	 args  => ['-XZ', '--in=../inputs/saccadetest.mgz',
		   '--out=mgz:mgz', '--textlog=test.txt',
		   '--input-frames=0-MAX@30',
		   '--output-frames=0-MAX@30',
		   '--boring-delay=max', '--movie',
		   '--ior-type=None', '--maxnorm-type=FancyOne',
		   '--display-map-factor=65000',
		   '--nodisplay-interp-maps',
		   '--display-head', '--ehc-type=Simple',
		   '--esc-type=Threshfric --hsc-type=Friction',
		   '--initial-eyepos=-1,-1','--initial-headpos=-2,-2',
		   '--nowta-sacsupp', '--nosalmap-sacsupp'],
	 files => ['test.txt', 'mgzT'],
     },

     {
	 name  => "ezlong-moviesurprise--1",
	 args  => ['-KZ', '--in=../inputs/ezframe#.pnm',
		   '--out=mgz:mgz', '--textlog=test.txt',
		   '--input-frames=0-5@30',
		   '--output-frames=0-MAX@30',
		   '--boring-delay=max', '--movie',
		   '--ior-type=None', '--surprise',
		   '--nodisplay-interp-maps'],
	 files => ['test.txt', 'mgzT'],
     },

     {
	 name  => "ezlong-moviesurprise--2",
	 args  => ['-KZ', '--in=../inputs/ezframe#.pnm',
		   '--out=mgz:mgz', '--textlog=test.txt',
		   '--input-frames=0-5@30',
		   '--output-frames=0-MAX@30',
		   '--boring-delay=max', '--movie',
		   '--ior-type=None', '--surprise',
		   '--nodisplay-interp-maps',
		   '--vc-chans=C:0.8333F:0.4167I:0.4167O:1.6667M:1.6667'],
	 files => ['test.txt', 'mgzT'],
     },

     );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/ezvision", @tests);
