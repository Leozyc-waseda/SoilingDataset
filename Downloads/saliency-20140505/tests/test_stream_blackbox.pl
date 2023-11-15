#!/usr/bin/perl -w

# $Id: test_stream_blackbox.pl 9547 2008-03-28 23:32:43Z rjpeters $

# This Perl script is a "black box" test driver. The idea is that we
# run the executable with different sets of command-line options, and
# compare the output files with reference files that are stored in the
# ref/ subdirectory. See blackbox.pm for implementation details.

use strict;

use blackbox;
use invt_config;

my @yuv_refs = ();
my @ppm_refs = ();

for (my $i = 0; $i < 50; $i++) {
    push(@yuv_refs, sprintf('mpgconverted%06d.640x480.yuv420p', $i));
    push(@ppm_refs, [sprintf('mpgconverted%06d.pnm', $i),
		     sprintf('mpgconverted%06d.ppm', $i)]);
}

# here is the key to the test suite: the database of command-line
# options along with the expected output files for those option sets

my @tests =
    (
     {
	 name  => 'raster-number-width--0',
	 args  => ['--test-mode',
		   '--in=../inputs/berry##.png',
		   '--input-frames=1-MAX@30Hz',
		   '--out=info:info.txt',
		   'berrycopy'],
	 files => ['info.txt'],
     },

     {
	 name  => 'raster-number-width--1',
	 args  => ['--test-mode',
		   '--in=../inputs/rgb-24-bit-#3#.png',
		   '--input-frames=1-MAX@30Hz',
		   '--out=info:info.txt',
		   'rgb-24-bit-copy'],
	 files => ['info.txt'],
     },

     {
	 name  => 'raster-number-width--2',
	 args  => ['--test-mode',
		   '--in=../inputs/rgb-24-bit-#3#.png',
		   '--input-frames=1-MAX@30Hz',
		   '--out=pnm:f#4#oo',
		   '--out=info:info.txt',
		   'rgb-24-bit-copy'],
	 files => ['foo-rgb-24-bit-copy0000.pnm', 'info.txt'],
     },

     # read a movie without crashing even when frame stride != 1
     {
	 name  => 'mpeg-stride--0',
	 args  => ['--test-mode',
		   '--in=../inputs/mpegclip1.mpg',
		   '--out=info:info.txt',
		   'mpeg-stride'],
	 files => ['info.txt'],
     },

     {
	 name  => 'mpeg2yuv-convertmovie--0',
	 args  => ['--test-mode',
		   '--in=../inputs/mpegclip1.mpg',
		   '--out=rawvideo',
		   '--out=info:info.txt',
		   'mpgconverted'],
	 files => [@yuv_refs, 'info.txt'],
     },

     # note we are using --out=bkg-rawvideo instead of just
     # --out=rawvideo here, to test DiskDataStream
     {
	 name  => 'mpeg2yuv-convertmovie--1',
	 args  => ['--test-mode',
		   '--in=../inputs/mpegclip2.mpg',
		   '--out=bkg-rawvideo',
		   'mpgconverted'],
	 files => [@yuv_refs],
     },

     # note, here we are testing --in=buf with a small buffer size
     {
	 name  => 'mpeg2yuv-convertmovie--2',
	 args  => ['--test-mode',
		   '--in=buf:../inputs/mpegclip1.mpg',
		   '--input-buffer-size=1',
		   '--underflow-strategy=retry',
		   '--out=ppm',
		   'mpgconverted'],
	 files => [@ppm_refs],
     },

     # note, here we are testing --in=buf with a large buffer size,
     # and also chaining together with a deinterlacer
     {
	 name  => 'mpeg2yuv-convertmovie--3',
	 args  => ['--test-mode',
		   '--in=buf:thf:../inputs/mpegclip2.mpg',
		   '--input-buffer-size=1000',
		   '--out=ppm',
		   'mpgconverted'],
	 files => [@ppm_refs],
     },

     {
	 name  => 'bob-deinterlacer--1',
	 args  => ['--test-mode',
		   '--in=bob:../inputs/mpegclip2.mpg',
		   '--input-frames=0-4-12@1Hz',
		   '--out=rawvideo',
		   'deinterlaced'],
	 files => ['deinterlaced000000.640x480.yuv420p',
		   'deinterlaced000001.640x480.yuv420p',
		   'deinterlaced000002.640x480.yuv420p',
		   'deinterlaced000003.640x480.yuv420p'],
     },

     {
	 name  => 'buffered-input--1',
	 args  => ['--test-mode',
		   '--in=buf:../inputs/ezframe#.pnm',
		   '--input-frames=0-2@1Hz',
		   '--input-buffer-size=1000',
		   '--out=hash:hash.txt',
		   'buffered-input'],
	 files => ['hash.txt'],
     },

     {
	 name  => 'plaintext-output--1',
	 args  => ['--test-mode',
		   '--in=../inputs/ezframe#.pnm',
		   '--input-frames=0-2@1Hz',
		   '--out=txt',
		   '--rescale-output=20x15',
		   'plaintext-output'],
	 files => ['plaintext-output000000.txt',
		   'plaintext-output000001.txt',
		   'plaintext-output000002.txt'],
     },

     {
	 name  => 'ccode-output--1',
	 args  => ['--test-mode',
		   '--in=../inputs/ezframe#.pnm',
		   '--input-frames=0-2@1Hz',
		   '--out=ccode',
		   '--rescale-output=20x15',
		   'ccode-output'],
	 files => ['ccode-output000000.C',
		   'ccode-output000001.C',
		   'ccode-output000002.C'],
     },

     {
         name  => 'yuv2ppm-convertframe--1',
         args  => ['--test-mode',
		   '--in=../inputs/frame#.yuv420p', '--input-frames=0-1000-8000@30Hz',
                   '--yuv-dims=640x480', '--out=pnm', '--output-frames=0-1000-8000@30Hz',
		   '--yuv-dims-loose',
                   'yuvconverted'],
         files => ['yuvconverted000000.pnm',
		   'yuvconverted001000.pnm',
		   'yuvconverted002000.pnm',
		   'yuvconverted003000.pnm',
		   'yuvconverted004000.pnm',
		   'yuvconverted005000.pnm',
		   'yuvconverted006000.pnm',
		   'yuvconverted007000.pnm',
		   'yuvconverted008000.pnm'],
     },
     );


# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::exec_prefix/bin/stream", @tests);
