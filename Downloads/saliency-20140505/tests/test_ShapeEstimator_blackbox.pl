#!/usr/bin/perl -w

# $Id: test_ShapeEstimator_blackbox.pl 10982 2009-03-05 05:11:22Z itti $

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
	 name  => "shape-estimator--1",
	 args  => ['-KZ', '--shape-estim-mode=FeatureMap',
		   '--in=../inputs/aircrafts.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=Fancy', '--textlog=shape.txt'],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm'],
     },

     {
	 name  => "shape-estimator--2",
	 args  => ['-KZ', '--shape-estim-mode=SaliencyMap',
		   '--in=../inputs/balloons.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=FancyOne', '--textlog=shape.txt'],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm',
		   'T000002.pnm', 'T000003.pnm',
		   'T000004.pnm', 'T000005.pnm'],
     },

     {
	 name  => "shape-estimator--3",
	 args  => ['-KZ', '--shape-estim-mode=FeatureMap',
		   '--in=../inputs/elephants.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=FancyWeak', '--vc-chans=CIO',
		   '--textlog=shape.txt', '--nodisplay-interp-maps'],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm',
		   'T000002.pnm', 'T000003.pnm'],
     },

     {
	 name  => "shape-estimator--4",
	 args  => ['-KZ', '--shape-estim-mode=ConspicuityMap',
		   '--in=../inputs/helis.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=FancyVWeak', '--vc-chans=CIO',
		   '--textlog=shape.txt'],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm',
		   'T000002.pnm', 'T000003.pnm',
		   'T000004.pnm', 'T000005.pnm'],
     },

     {
	 name  => "shape-estimator--5",
	 args  => ['-KZ', '--shape-estim-mode=SaliencyMap',
		   '--in=../inputs/faces.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=Fancy', '--vc-chans=CIO',
		   '--textlog=shape.txt'],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm',
		   'T000002.pnm'],
     },

     {
	 name  => "shape-estimator--6",
	 args  => ['-KZ', '--shape-estim-mode=FeatureMap',
		   '--in=../inputs/sailboats.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=Fancy', '--textlog=shape.txt',
		   '--vc-chans=CIO', '--noshape-estim-largeneigh'],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm',
		   'T000002.pnm', 'T000003.pnm',
		   'T000004.pnm', 'T000005.pnm'],
     },

     {
	 name  => "shape-estimator--7",
	 args  => ['-KZ', '--shape-estim-mode=FeatureMap',
		   '--in=../inputs/faces.ppm', '--out=pnm: ',
		   '--output-frames=@EVENT', '-+', '--too-much-time=700ms',
		   '--maxnorm-type=FancyOne', '--textlog=shape.txt',
		   '--vc-chans=CIOLTXEK', '--mega-combo-zoom=8',
		   '--nomega-combo-topcm', '--nodisplay-interp-maps' ],
	 files => ['shape.txt', 'T000000.pnm', 'T000001.pnm',
		   'T000002.pnm' ],
     },

     );

# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/ezvision", @tests);
