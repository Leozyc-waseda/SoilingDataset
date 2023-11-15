#!/usr/bin/perl -w

# $Id: test_ezvision_blackbox.pl 11159 2009-05-02 03:04:10Z itti $

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

     # this test uses the reference outputs created by the test of the same name but running ezvision, in
     # test_ezvision_blackbox.pl; this allows us to ensure that simple-saliency used here generates exactly the same
     # outputs as ezvision:
     {
	 name  => "ez-simplesaliency--1",
	 args  => ['../inputs/ezin T VCO' ],
	 files => [ 'T000000.png', 'T000001.png', 'T000002.png', 'T000003.png', 'T000004.png',
		    'VCO000000.pfm', 'VCO000001.pfm', 'VCO000002.pfm', 'VCO000003.pfm', 'VCO000004.pfm' ],
     },

    );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/simple-saliency", @tests);
