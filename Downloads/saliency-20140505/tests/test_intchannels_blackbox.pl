#!/usr/bin/perl -w

# $Id: test_intchannels_blackbox.pl 7861 2007-02-07 23:47:04Z rjpeters $

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
	 name  => "intchannels-conservative--1",
	 args  => ['--in=../inputs/mpegclip1.mpg', '--input-frames=0-2@30Hz',
		   '--out=ppm:',
		   '--int-chan-scale-bits=30',
		   '--int-math-lowpass5=lp5std',
		   '--int-math-lowpass9=lp9std',
		   '--save-channel-outputs', '--save-vcx-output'],
	 files => ['ICOint-color-000000.pnm', 'ICOint-color-000001.pnm', 'ICOint-color-000002.pnm',
		   'ICOint-motion-000000.pnm', 'ICOint-motion-000001.pnm', 'ICOint-motion-000002.pnm',
		   'ICOint-orientation-000000.pnm', 'ICOint-orientation-000001.pnm', 'ICOint-orientation-000002.pnm',
		   'ISOint-by-000000.pnm', 'ISOint-by-000001.pnm', 'ISOint-by-000002.pnm',
		   'ISOint-dir_0-000000.pnm', 'ISOint-dir_0-000001.pnm', 'ISOint-dir_0-000002.pnm',
		   'ISOint-dir_1-000000.pnm', 'ISOint-dir_1-000001.pnm', 'ISOint-dir_1-000002.pnm',
		   'ISOint-dir_2-000000.pnm', 'ISOint-dir_2-000001.pnm', 'ISOint-dir_2-000002.pnm',
		   'ISOint-dir_3-000000.pnm', 'ISOint-dir_3-000001.pnm', 'ISOint-dir_3-000002.pnm',
		   'ISOint-flicker-000000.pnm', 'ISOint-flicker-000001.pnm', 'ISOint-flicker-000002.pnm',
		   'ISOint-intensity-000000.pnm', 'ISOint-intensity-000001.pnm', 'ISOint-intensity-000002.pnm',
		   'ISOint-ori_0-000000.pnm', 'ISOint-ori_0-000001.pnm', 'ISOint-ori_0-000002.pnm',
		   'ISOint-ori_1-000000.pnm', 'ISOint-ori_1-000001.pnm', 'ISOint-ori_1-000002.pnm',
		   'ISOint-ori_2-000000.pnm', 'ISOint-ori_2-000001.pnm', 'ISOint-ori_2-000002.pnm',
		   'ISOint-ori_3-000000.pnm', 'ISOint-ori_3-000001.pnm', 'ISOint-ori_3-000002.pnm',
		   'ISOint-rg-000000.pnm', 'ISOint-rg-000001.pnm', 'ISOint-rg-000002.pnm',
		   'IVCO000000.pnm', 'IVCO000001.pnm', 'IVCO000002.pnm'],
	 },

     {
	 name  => "intchannels-aggressive--1",
	 args  => ['--in=../inputs/mpegclip1.mpg', '--input-frames=0-2@30Hz',
		   '--out=ppm:',
		   '--int-chan-scale-bits=16',
		   '--int-math-lowpass5=lp5optim',
		   '--int-math-lowpass9=lp9optim',
		   '--save-channel-outputs', '--save-vcx-output'],
	 files => ['ICOint-color-000000.pnm', 'ICOint-color-000001.pnm', 'ICOint-color-000002.pnm',
		   'ICOint-motion-000000.pnm', 'ICOint-motion-000001.pnm', 'ICOint-motion-000002.pnm',
		   'ICOint-orientation-000000.pnm', 'ICOint-orientation-000001.pnm', 'ICOint-orientation-000002.pnm',
		   'ISOint-by-000000.pnm', 'ISOint-by-000001.pnm', 'ISOint-by-000002.pnm',
		   'ISOint-dir_0-000000.pnm', 'ISOint-dir_0-000001.pnm', 'ISOint-dir_0-000002.pnm',
		   'ISOint-dir_1-000000.pnm', 'ISOint-dir_1-000001.pnm', 'ISOint-dir_1-000002.pnm',
		   'ISOint-dir_2-000000.pnm', 'ISOint-dir_2-000001.pnm', 'ISOint-dir_2-000002.pnm',
		   'ISOint-dir_3-000000.pnm', 'ISOint-dir_3-000001.pnm', 'ISOint-dir_3-000002.pnm',
		   'ISOint-flicker-000000.pnm', 'ISOint-flicker-000001.pnm', 'ISOint-flicker-000002.pnm',
		   'ISOint-intensity-000000.pnm', 'ISOint-intensity-000001.pnm', 'ISOint-intensity-000002.pnm',
		   'ISOint-ori_0-000000.pnm', 'ISOint-ori_0-000001.pnm', 'ISOint-ori_0-000002.pnm',
		   'ISOint-ori_1-000000.pnm', 'ISOint-ori_1-000001.pnm', 'ISOint-ori_1-000002.pnm',
		   'ISOint-ori_2-000000.pnm', 'ISOint-ori_2-000001.pnm', 'ISOint-ori_2-000002.pnm',
		   'ISOint-ori_3-000000.pnm', 'ISOint-ori_3-000001.pnm', 'ISOint-ori_3-000002.pnm',
		   'ISOint-rg-000000.pnm', 'ISOint-rg-000001.pnm', 'ISOint-rg-000002.pnm',
		   'IVCO000000.pnm', 'IVCO000001.pnm', 'IVCO000002.pnm'],
	 },
     );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/test-intVisualCortex", @tests);
