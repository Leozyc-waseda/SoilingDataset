#!/usr/bin/perl -w

# $Id: testlong_movies_blackbox.pl 10982 2009-03-05 05:11:22Z itti $

# This Perl script is a "black box" test driver. The idea is that we
# run the executable with different sets of command-line options, and
# compare the output files with reference files that are stored in the
# ref/ subdirectory. See blackbox.pm for implementation details.

use strict;
use Cwd;
use blackbox;
use invt_config;

sub build_frame_list {
    my $stem = shift;
    my $first = shift;
    my $last = shift;

    my @result = ();

    for (my $i = $first; $i <= $last; $i++) {
	push @result, [sprintf("%s%06d.pnm", $stem, $i),
		       sprintf("%s%06d.ppm", $stem, $i)];
    }

    return @result;
}

# here is the key to the test suite: the database of command-line
# options along with the expected output files for those option sets

my $infile = "$invt_config::abs_srcdir/inputs/testmovie001.tbz";

my @opt0 = ("--debug=no", "--textlog=testmovie001.foa");
my @opt1 = ("-XZ", @opt0, "--nodisplay-additive",
	    "--ior-type=None", "--display-patch");
my @opt2 = (@opt1, "--foveate-input-depth=5", "--trm-type=KillStatic");
my @noeye = ("--boring-delay=forever", "--boring-sm-mv=0.0",
	     "--nodisplay-eye", "--nodisplay-eye-traj");

my @ref_files = ("testmovie001.foa", build_frame_list("rawframe", 0, 99));

my @tests =
    (
     {
	 name  => "movie-traj--1",
	 args  => ["-TZ", @opt0, @noeye, "--vc-chans=CIO", $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-traj--2",
	 args  => ["-XZ", @opt0, @noeye, "--vc-chans=CIOF", $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-traj--3",
	 args  => ["-XZ", @opt0, @noeye, $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-traj--4",
	 args  => ["-XZ", @opt0, @noeye, "--nodisplay-additive", $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-traj--5",
	 args  => [@opt1, @noeye, $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-foveate--1",
	 args  => [@opt1, @noeye, "--foveate-input-depth=5", $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-foveate--2",
	 args  => [@opt1, @noeye, "--foveate-input-depth=2",
		   $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-killstatic-1",
	 args  => [@opt2, @noeye, $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-saccade--1",
	 args  => [@opt2, "--ehc-type=Simple",
		   "--hsc-type=None --esc-type=Fixed",
		   "--initial-eyepos=-2,-2", $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-saccade--2",
	 args  => [@opt2, "--ehc-type=Simple",
		   "--hsc-type=None --esc-type=Friction",
		   "--initial-eyepos=-2,-2",  $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-saccade--3",
	 args  => [@opt2, "--ehc-type=Simple",
		   "--hsc-type=None --esc-type=Threshold", $infile],
	 files => [@ref_files],
     },

     {
	 name  => "movie-saccade--4",
	 args  => [@opt2, "--ehc-type=Simple --hsc-type=None",
		   "--esc-type=Threshfric", $infile],
	 files => [@ref_files],
     },

     );

# Run the black box tests; note that the default executable can be
# overridden from the command-line with "--executable"

blackbox::run("$invt_config::abs_top_srcdir/scripts/process_movie.pl "
	      . "--rawframes --testmode --notmpdir "
	      . "--vision-exec=$invt_config::exec_prefix/bin/ezvision",
	      @tests);
