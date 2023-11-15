#!/usr/bin/perl -w

# $Id: test_attention_gate_blackbox.pl 11967 2009-02-13 09:40:54Z itti $

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
	 name  => "ag-basic--1",
	 args  => ['--ior-type=None', '--agm-type=Std',
		   '--surprise', '--vc-chans=OLTXEH', '--nouse-random', 
		   '--display-foa', '--ior-type=None', '--sm-type=Trivial',
		   '--wta-type=None', '--in=../inputs/ag/stim15_046_#.png',
		   '--input-frames=0-MAX@50ms', '--output-frames=0-MAX@50ms',
		   '--out=png:stim', '--sv-type=Stats', '--sv-stats-fname=agstats.txt', 
		   '--save-channel-stats', '--save-channel-stats-name=agchan.txt', 
		   '--save-channel-stats-tag=stim15_046', '--save-stats-per-channel',
		   '--save-ag', '--ag-type=Std' ,'--savestats-ag', '--ag-statsfile=ag.stats.txt',
		   '--ag-tframe=0', '--ag-maskfile=../inputs/ag/mask.png'],
	 files => ['stim-AG000019.png',
		   'stim-AG-CAM000001.png',
		   'stim-AG-LMASK000013.png',
		   'stim-AG-STAT-MASK000031.png',
		   'stim-AG000016.png',
		   'stim-AG-CAM000016.png',
		   'stim-AG-LMASK000016.png',
		   'stim-AG-STAT-MASK000016.png',
		   'agchan.txt.final-AGmask.txt']
     }
    );



# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/ezvision", @tests);
