# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/tests/blackbox.pm $
# $Id: blackbox.pm 11159 2009-05-02 03:04:10Z itti $

# This Perl file contains shared routines for "black-box" test
# scripts. Black-box tests are tests of an actual production
# executable (i.e. one that does "real work"), in constrast to
# white-box tests, which use an executable whose only purpose is to
# run tests. With black-box tests, we drive the executable with
# various sets of command-line options, parameter files, etc., and
# compare its output(s) with known correct reference output(s) that
# are stored in the ${REF_DIR}/ subdirectory.

# One caveat is that changes to the floating-point calls in C++ may
# lead to small differences in the output (often this shows up as a
# single-bit difference, with a byte that was previously 254 now being
# 255, or vice versa). In these cases, the easiest way to look for
# differences is with

# "cmp -l [file1] [file2]"

# which will print a list of the offending bytes, and their values in
# the two files. If the differences are minor enough to be reasonably
# due to simple changes in the way that the floating-point code was
# compiled, then once can replace the old reference file with the new
# one.

package blackbox;

require Exporter;

@ISA = qw(Exporter);

use strict;
use File::Basename;
use File::Copy;

use testsuite;

$blackbox::OPT_PATTERN = "";
$blackbox::OPT_INTERACTIVE = 0;
$blackbox::OPT_CREATE_REF = 0;
$blackbox::OPT_REPLACE_REF = 0;
$blackbox::OPT_LIST_TESTS = 0;
$blackbox::OPT_LIST_REFS = 0;
$blackbox::OPT_CLEANUP = 1;
$blackbox::REF_DIR = "$invt_config::abs_srcdir/ref";
$blackbox::executable = "(none)";
$blackbox::cmpprog = "cmp"; # i.e., /usr/bin/cmp
$blackbox::EXTRA_ARGS = "";
$blackbox::RUN_DIR = "$invt_config::abs_srcdir/blackbox.run.$$";
$blackbox::ROOT_DIR = "$invt_config::abs_srcdir";

@blackbox::cleanup_code = ();

# If true, then don't actually compare the reference and generated
# files. This is useful only for using the test suite as a
# benchmark suite -- running and summarizing the detailed file
# comparisons takes a variable amount of time depending on how
# different the files are, so in order to use the test suite for
# benchmarking, we want to be able to skip the comparisons.
$blackbox::OPT_NO_COMPARISON = 0;

%blackbox::options =
    (
     "--help"          =>  { varref => \$testsuite::SHOW_HELP,
			     optval => 1,
			     help => "print this help listing" },
     "--debug"         =>  { varref => \$testsuite::OPT_DEBUG,
			     optval => 1,
			     help => "print debugging information about the test suite perl framework itself" },
     "--list"          =>  { varref => \$blackbox::OPT_LIST_TESTS,
			     optval => 1,
			     help => "list available tests" },
     "--list-refs"     =>  { varref => \$blackbox::OPT_LIST_REFS,
			     optval => 1,
			     help => "list reference files" },
     "--executable"    =>  { varref => \$blackbox::executable,
			     help => "override the default executable with a path to a different executable to be tested" },
     "--verbosity"     =>  { varref => \$testsuite::VERBOSITY,
			     help => "specify a different verbosity level:\n* -1 = no output, result of test run is given by exit status;\n* 0 = just print a single summary line indicating how many tests succeeded;\n* 1 = just print one line per test indicating success or failure;\n* 2 = print the command-line options used for each test, and list each reference file that is tested;\n* 3 = as before, but print detailed analysis of any failed reference tests;\n* 4 = as before, but show the stdout+stderr from every command, even if the test doesn't fail" },
     "--match"         =>  { varref => \$blackbox::OPT_PATTERN,
			     help => "only consider tests matching this regexp" },
     "--interactive"   =>  { varref => \$blackbox::OPT_INTERACTIVE,
			     help => "specify an interactivity level (0, 1, or 2): when a test comparison fails, ask the user whether to replace the existing reference file with the current test files" },
     "--quit-on-fail"  =>  { varref => \$testsuite::OPT_QUIT_ON_FAIL,
			     help => "quit immediately if any test fails" },
     "--refdir"        =>  { varref => \$blackbox::REF_DIR,
			     help => "specify an alternate directory containing reference files; combined with --createref, this can be useful to test non-standard build architectures or compile options: first use pristine sources to generate a new set of reference files using --refdir myref --createref, then modify the sources, then verify that the tests still pass using just --refdir myref" },
     "--createref"     =>  { varref => \$blackbox::OPT_CREATE_REF,
			     optval => 1,
			     help => "instantiate non-existent reference files from the test files: if any reference files are missing, they will be created from the corresponding current test file" },
     "--replaceref"    =>  { varref => \$blackbox::OPT_REPLACE_REF,
			     optval => 1,
			     help => "USE GREAT CARE WITH THIS OPTION! for any test file that is found not to match its reference file, automatically overwrite the non-matching reference file with current test file" },
     "--nocomparison"  =>  { varref => \$blackbox::OPT_NO_COMPARISON,
			     optval => 1,
			     help => "skip reference file comparisons; this is useful for benchmarking: if you run the suite with --nocomparison, the cpu time will be predominantly spent just in running the tests, so for example you could compare the times across different build options" },
     "--noprofiling"   =>  { varref => \$testsuite::DO_PROFILING,
			     optval => 0,
			     help => "skip parsing of gmon.out profile files that may be generated by the executables being tested" },
     "--save-gmons"    =>  { varref => \$testsuite::SAVE_GMONS,
			     optval => 1,
			     help => "don't remove any gmon.out files that may be generated by the executable being tested" },
     "--extra-args"    =>  { varref => \$blackbox::EXTRA_ARGS,
			     help => "give additional arguments (e.g., for debugging) to be passed to the executable being tested when each test is run" },
     "--nocleanup"    =>  { varref => \$blackbox::OPT_CLEANUP,
			     optval => 0,
			     help => "don't remove the sandbox directory in which the test is run; this may be helpful for debugging" },
     );

# Check if a given filename has a recognized image-file extension

# usage: is_image_file($fname)
sub is_image_file {
    my $fname = $_[0];

    if    ($fname =~ /\.ppm$/) { return 1; }
    elsif ($fname =~ /\.pgm$/) { return 1; }
    elsif ($fname =~ /\.pbm$/) { return 1; }
    elsif ($fname =~ /\.pnm$/) { return 1; }
    elsif ($fname =~ /\.pfm$/) { return 1; }
    elsif ($fname =~ /\.png$/) { return 1; }

    return 0;
}

# Give the user a query message, and optionally view a set of (image)
# files, and return the user's response to the query. Currently we
# just use "xv" to display the files, assuming they are images; we
# could eventually do something more intelligent for cases where they
# are non-image files, although even in this case, xv is ok because it
# pops up a text editor window.

# usage: view_files_and_query($query_msg, @images)
sub view_files_and_query {
    my $query_msg = shift;
    my @images = @_;

    my @pids = ();

    if ($blackbox::OPT_INTERACTIVE >= 2) {
        my $xpos = 20;
        my $ypos = 20;

        # FIXME It would be nice to get the exact width to
        # separate the various images, rather than hardcoding 300
        # here:
        my $xstep = 300;

        foreach my $img (@images) {
            my $geom = "+${xpos}+${ypos}";
	    vputs(">=", 2, "\n@ ${geom}: $img");
	    if (my $pid = fork) {
		# parent
		push @pids, $pid;
	    }
	    elsif (defined $pid) {
		# child
		exec "xv", "-geometry", $geom, $img;
	    }
	    else {
		die "\nCan't fork: $!\n";
	    }

            $xpos += $xstep;
        }
    }

    # collect response from user:
    my $response = "";
    do {
	vputs(">=", 0, "\n$query_msg\n");
	$response = <STDIN>; chomp $response;
    } while ($response ne "y" && $response ne "n");

    kill 'TERM', @pids;

    return $response
}


# Give the user a query message, view a pair of image files using
# bin/imageCompare from this toolkit, and return the user's response
# to the query

# usage: compare_files_and_query($query_msg, $img1, $img2)
sub compare_files_and_query {
    my $query_msg = shift;
    my $img1 = shift;
    my $img2 = shift;

    my @pids = ();

    my $imagecmp = "$invt_config::exec_prefix/bin/imageCompare";

    if (-x $imagecmp && $blackbox::OPT_INTERACTIVE >= 2
	&& is_image_file($img1) && is_image_file($img2)) {

	if (my $pid = fork) {
	    # parent
	    push @pids, $pid;
	}
	elsif (defined $pid) {
	    # child
	    exec $imagecmp, $img1, $img2;
	}
	else {
	    die "\nCan't fork: $!\n";
	}
    }

    # collect response from user:
    my $response = "";
    do {
	vputs(">=", 0, "\n$query_msg\n");
	$response = <STDIN>; chomp $response;
    } while ($response ne "y" && $response ne "n");

    kill 'TERM', @pids;

    return $response;
}

# usage: has_textfile_extension($fname)
sub has_textfile_extension {
    my $fname = $_[0];

    if ($fname =~ /\.txt$/) { return 1; }
    if ($fname =~ /\.foa$/) { return 1; }
    if ($fname =~ /\.pmap$/) { return 1; }
    if ($fname =~ /\.stsd$/) { return 1; }
    if ($fname =~ /\.(c|h|cc|hh|cpp|hpp|C|H)$/) { return 1; }

    return 0;
}

# If the given file is not already gzipped, then gzip it. In any case,
# return the name of the gzipped file.

# usage: gzip_if_needed($filename)
sub gzip_if_needed {

    my $filename = $_[0];

    # If the filename already ends in ".gz", then leave it as
    # is. Otherwise, if the filename ends in an extension that
    # implies a text format, then also leave it as is (that way we
    # can get text diffs in the version control
    # history). Otherwise, assume that it's a binary file and gzip
    # it.

    if ($filename =~ /\.gz$/) {
	# ok, it's already gzipped; leave as is
	return $filename;
    }
    elsif (has_textfile_extension($filename)) {
	# ok, it's a text file, so don't gzip it
	return $filename;
    }
    else {
	system("gzip", "-9f", $filename) == 0
	    or die "\ngzip failed: $?\n";
	return "${filename}.gz";
    }
}

# If the given file is gzipped, then gunzip it. In any case, return
# the name of the uncompressed file.

# usage: gunzip_if_needed($filename)
sub gunzip_if_needed {

    my $filename = $_[0];

    if ($filename =~ /(.*)\.gz$/) {
	$filename = $1;
	system("gzip", "-d", "--force", "${filename}.gz");
	unlink "${filename}.gz";
    }

    return $filename;
}

# Return the reference file and test file names from a given file
# specification. The filespec should be either a string, or an
# anonymous list with two elements, one the name of the test file, and
# one the name of the corresponding ref file.

# usage: get_test_and_ref_names($test_name, $filespec)
sub get_test_and_ref_names {

    my $test_name = $_[0];
    my $filespec = $_[1];

    my $test_file;
    my $ref_file;

    # the default is to look in the "${REF_DIR}/"
    # subdirectory for a reference file against which to
    # compare the current file to be checked...
    if (not ref $filespec) {
	$test_file = $filespec;
	$ref_file = "${blackbox::REF_DIR}/${test_name}--$filespec";
    }
    elsif (ref($filespec) eq "ARRAY") {

	# ...but if there is a different file specified in the
	# options:

        $test_file = $filespec->[0];

        my $new_name = $filespec->[1];

        # if this new name is a directory, then use it to
        # point to the ref file
	if ($new_name =~ m|/$|) {
            $ref_file = "${new_name}${test_name}--${test_file}";
        }
	else {

            # but otherwise just use the the new name as
            # an alternate filename in the default ref
            # directory
            $ref_file = "${blackbox::REF_DIR}/${test_name}--${new_name}";
        }
    }

    return ($test_file, $ref_file)
}

# Creates a gzipped reference file from the given test file. Returns
# the name of the gzipped ref file.

# usage: create_ref_file($test_file, $ref_file)
sub create_ref_file {

    my $test_file = $_[0];
    my $ref_file = $_[1];

    # This creates the directory to contain the ref file if it
    # does not yet exist. This is just a silent no-op if the
    # directory already exists.
    my $dir = dirname($ref_file);
    if (! -d $dir) {
	mkdir($dir, 0755)
	    or die "\ncouldn't create directory $dir: $!\n";
    }

    copy($test_file, $ref_file);

    return gzip_if_needed($ref_file);
}

# Look up a reference file matching a given name. If the exact
# reference file doesn't exist, (1) try looking for its gzipped twin,
# or (2) optionally create a reference file from the current test
# file, or else (3) give up. Returns the name of the matching
# reference file, or the empty string if no reference file is found.

# usage: lookup_ref_file($ref_file, $test_file)
sub lookup_ref_file {
    my $ref_file = $_[0];
    my $test_file = $_[1];

    if (-f $ref_file) { return $ref_file; }
    if (-f "${ref_file}.gz") { return "${ref_file}.gz"; }

    if ( $blackbox::OPT_CREATE_REF ) {
        vputs(">=", 2, "(creating reference file from results) ");

        return create_ref_file($test_file, $ref_file);

    }
    elsif ( $blackbox::OPT_INTERACTIVE ) {

        my $response =
            view_files_and_query
	    ("create reference file from results (y or n)?",
	     $test_file);

        if ($response eq "y") {

            return create_ref_file($test_file, $ref_file);
        }
    }

    # if we get here, all other attempts have failed, so...

    vputs(">=", 2, "FAILED!\n\treference file '$ref_file' is missing!");

    return "";
}

# Make a copy of filename, gunzipping it in the process if necessary,
# so that the resulting copy is unequivocally uncompressed.

# usage: make_gunzipped_copy($filename)
sub make_gunzipped_copy {
    my $filename = $_[0];

    my $copyname = basename($filename);

    if (-f "./$copyname") { unlink "./$copyname"; }

    copy($filename, "./$copyname");

    return gunzip_if_needed($copyname);
}

# Replace an existing ref file with a new one; we still keep the old
# ref file around though in a temporary location, so that we can
# examine it after the test script is finished.

# usage: replace_ref_file($current_ref_file, $new_ref_file)
sub replace_ref_file {
    my $current_ref_file = $_[0];
    my $new_ref_file = $_[1];

    # Gzip the new ref file to match the compression state of the
    # existing ref file, if necessary
    if ($current_ref_file =~ /\.gz$/) {
        system("gzip", "-9", "--force", $new_ref_file) == 0
	    or die "\ngzip failed: $?\n";

        $new_ref_file = "${new_ref_file}.gz";
    }

    my $ref_tail = basename($current_ref_file);

    # Check that the names are now actually the same, including
    # any gzipping and whatnot
    if ($ref_tail ne basename($new_ref_file)) {
        die "\nnew and old reference file names don't match:"
            . "$new_ref_file and $current_ref_file\n";
    }

    # It's possible that we're here because the ref file doesn't
    # exist at all yet, so we have to check for that...
    if (-f $current_ref_file) {

        # ...if it does exist, then move the current ref file
        # aside (but don't remove it outright, so that we still
        # have a chance to inspect it after the script has been
        # run)
	my $dest = "./previous-$ref_tail";
	if (-f $dest) { unlink $dest; }

        rename($current_ref_file, $dest);
    }

    # And finally put the new ref file in place
    rename($new_ref_file, $current_ref_file);
}

# Here we expect 'cmp_l_results' to be the stdout resulting from
# running "cmp -l" to compare two files

# usage: show_cmp_l_stats($file1size, $file2size, @cmp_l_results)
sub show_cmp_l_stats {
    my $file1size = shift;
    my $file2size = shift;
    my @cmp_l_results = @_;

    my $num_diffs = 0;
    my $sum_file_1 = 0;
    my $sum_file_2 = 0;
    my $sum_diffs = 0;
    my $sum_abs_diffs = 0;
    my $sum_offset_pos = 0;

    my @num_offset_pos_mod_2 = (0, 0);
    my @num_offset_pos_mod_3 = (0, 0, 0);
    my @num_offset_pos_mod_4 = (0, 0, 0, 0);

    my %diffs;

    foreach my $line (@cmp_l_results) {

	$line =~ s/^[ \t]+//;
	$line =~ s/[ \t]+$//;

	my @lineparts = split(/[ \t]+/, $line);

	if (scalar(@lineparts) != 3) {
	    die "\ncouldn't parse cmp -l line '$line'";
	}

	my $offset_pos = $lineparts[0];

	# "cmp -l" reports byte values as octal numbers, so we
	# need to use oct() to convert the octal back to decimal
	# before we try to do any arithmetic
	my $byte1 = oct($lineparts[1]);
	my $byte2 = oct($lineparts[2]);

	$num_diffs++;
	$sum_file_1 += $byte1;
	$sum_file_2 += $byte2;
	my $diff = $byte2 - $byte1;
	$sum_diffs += $diff;
	$sum_abs_diffs += abs($diff);
	if ( not defined $diffs{$diff}) {
	    $diffs{$diff} = 0;
	}
	$diffs{$diff}++;

	$sum_offset_pos += $offset_pos;

	$num_offset_pos_mod_2[$offset_pos % 2]++;
	$num_offset_pos_mod_3[$offset_pos % 3]++;
	$num_offset_pos_mod_4[$offset_pos % 4]++;
    }

    vputs(">=", 3, "\ncomparison statistics:");

    foreach my $diff (sort { $a <=> $b } keys(%diffs)) {
        vputs(">=", 3, "\n\tmagnitude $diff: $diffs{$diff} diffs");
    }

    vputs(">=", 3, "\n\tnum diff locations: ". $num_diffs);
    vputs(">=", 3, "\n\tfile1 length: " . $file1size . " bytes");
    vputs(">=", 3, "\n\tfile2 length: " . $file2size . " bytes");
    vputs(">=", 3, "\n\t% of bytes differing: " . ((100.0*$num_diffs)/$file1size));
    vputs(">=", 3, "\n\tmean offset position: " . ($sum_offset_pos/$num_diffs));
    vputs(">=", 3, "\n\tnum (file diff location % 2) == 0: " . $num_offset_pos_mod_2[0]);
    vputs(">=", 3, "\n\tnum (file diff location % 2) == 1: " . $num_offset_pos_mod_2[1]);
    vputs(">=", 3, "\n\tnum (file diff location % 3) == 0: " . $num_offset_pos_mod_3[0]);
    vputs(">=", 3, "\n\tnum (file diff location % 3) == 1: " . $num_offset_pos_mod_3[1]);
    vputs(">=", 3, "\n\tnum (file diff location % 3) == 2: " . $num_offset_pos_mod_3[2]);
    vputs(">=", 3, "\n\tnum (file diff location % 4) == 0: " . $num_offset_pos_mod_4[0]);
    vputs(">=", 3, "\n\tnum (file diff location % 4) == 1: " . $num_offset_pos_mod_4[1]);
    vputs(">=", 3, "\n\tnum (file diff location % 4) == 2: " . $num_offset_pos_mod_4[2]);
    vputs(">=", 3, "\n\tnum (file diff location % 4) == 3: " . $num_offset_pos_mod_4[3]);
    vputs(">=", 3, "\n\tsum of file1 bytes (at diff locations): " . $sum_file_1);
    vputs(">=", 3, "\n\tsum of file2 bytes (at diff locations): " . $sum_file_2);
    vputs(">=", 3, "\n\tmean diff (at diff locations): " . ($sum_diffs/$num_diffs));
    vputs(">=", 3, "\n\tmean abs diff (at diff locations): " . ($sum_abs_diffs/$num_diffs));
    vputs(">=", 3, "\n\tmean diff (at all locations): " . ($sum_diffs/$file1size));
    vputs(">=", 3, "\n\tmean abs diff (at all locations): " . ($sum_abs_diffs/$file1size));
}

# usage: run_comparison($file1, $file2)
sub run_comparison {

    my $file1 = $_[0];
    my $file2 = $_[1];

    my ($f1name, $f1path, $ext1) = fileparse($file1, '\..*');
    my ($f2name, $f2path, $ext2) = fileparse($file2, '\..*');

    # if the files are bzipped files, then use 'bzcmp' for
    # comparison instead of plain 'cmp'
    if ($ext1 eq $ext2 &&
	($ext1 eq ".tbz" || $ext1 eq ".bz2")) {

	my $file1_temp_copy = testsuite::tempfile_name();
	my $file2_temp_copy = testsuite::tempfile_name();

        system("bzcat $file1 > $file1_temp_copy") == 0
	    or die "\nbzcat failed: $?\n";

        system("bzcat $file2 > $file2_temp_copy") == 0
	    or die "\nbzcat failed: $?\n";

        $file1 = $file1_temp_copy;
        $file2 = $file2_temp_copy;
    }

    if (has_textfile_extension($file1) &&
	has_textfile_extension($file2)) {

	open CHAN, "diff -au $file2 $file1 |";
	my @diff_results = <CHAN>;
        close CHAN;

        vputs(">=", 3, "\ndiff results:\n\t");
        vputs(">=", 3, join("\t", @diff_results));

    } else {

        open CHAN, "$blackbox::cmpprog -l $file1 $file2 |";
	my @cmp_l_results = <CHAN>;

        close CHAN;

	if ($? != 0 && scalar(@cmp_l_results) == 0) {
	    vputs(">=", 3,
		  "cmp -l failed but didn't report any differences");
	}
	else {

	    if (scalar @cmp_l_results <= 10) {
		vputs(">=", 3, join("", @cmp_l_results));
	    }
	    else {

		my $file1size = (-s $file1);
		my $file2size = (-s $file2);

		show_cmp_l_stats($file1size, $file2size, @cmp_l_results);
	    }

	    my $corrcoef_prog = $invt_config::abs_top_builddir . "/bin/corrcoef";

	    if (-x $corrcoef_prog) {

		if (is_image_file($file1) && is_image_file($file2)) {

		    open CHAN, "$corrcoef_prog $file1 $file2 2> /dev/null |";

		    while (<CHAN>) {
			if (/^corrcoef.*=(.*)$/) {
			    vputs(">=", 3, "\n\tcorrcoef: "
				  . sprintf("%.6f", $1));
			}
		    }

		    close CHAN;
		}
	    }
	}
    }
}

# In verbose or interactive mode, we can provide some more detailed
# info about the differences between the two files, and possibly give
# the user a chance to replace the existing ref file.

# usage: do_detailed_comparison($test_file, $ref_file, $ref_file_copy)
sub do_detailed_comparison {

    my $test_file = $_[0];
    my $ref_file = $_[1];
    my $ref_file_copy = $_[2];

    vputs(">=", 3, "\n");

    run_comparison($test_file, $ref_file_copy);

    my @md5info1 = split(/[ \t]+/, `md5sum $test_file`);
    my @md5info2 = split(/[ \t]+/, `md5sum $ref_file_copy`);

    vputs(">=", 3, "\n\tmd5sum (test) ${test_file}:\n\t\t$md5info1[0]");
    vputs(">=", 3, "\n\tmd5sum (ref)  ${ref_file_copy}:\n\t\t$md5info2[0]");

    if ( ! -d "./trash" ) {
        mkdir("./trash");
    }

    if (-f "./trash/$ref_file_copy") {
	unlink "./trash/$ref_file_copy";
    }

    rename($ref_file_copy, "./trash/$ref_file_copy");

    if ( $blackbox::OPT_INTERACTIVE || $blackbox::OPT_REPLACE_REF ) {

	my $response;

        if ( $blackbox::OPT_REPLACE_REF ) {
            $response = "y";
            vputs(">=", 2, "\n(replacing $ref_file)");
        }
	else {
            $response =
		compare_files_and_query
		("replace previous reference file (y or n)?",
		 $test_file, "./trash/$ref_file_copy");
        }

        if ($response eq "y") {
            rename($test_file, $ref_file_copy);

            replace_ref_file($ref_file, $ref_file_copy);
        }
    }
}

# compare a test file with the reference file; returns 0 if
# everything was OK, otherwise returns non-zero
# usage: check_one_file($test_file, $ref_file)
sub check_one_file {
    my $test_file = $_[0];
    my $ref_file = $_[1];

    vputs(">=", 2, "\nchecking $test_file ... ");

    if ( ! -f $test_file ) {
        vputs(">=", 2, "FAILED!\n\ttest file '$test_file' is missing!");

        return 1;
    }

    $ref_file = lookup_ref_file($ref_file, $test_file);

    # Check if ref file lookup failed
    if (length($ref_file) == 0) { return 1; }

    # See if we want to skip the file comparison (e.g. if we're
    # using the test suite for benchmarking):
    if ( $blackbox::OPT_NO_COMPARISON ) {
        vputs(">=", 2, "comparison skipped for benchmarking");
        unlink($test_file);
        return 0;
    }

    # OK, we're not skipping the comparison, so let's proceed with
    # the detailed comparison:

    my $ref_file_copy = make_gunzipped_copy($ref_file);

    my $result = `diff -q $test_file $ref_file_copy`;
    my $code = $?;

    if ( $code == 0 ) {
        vputs(">=", 2, "ok");
        unlink($test_file);
        unlink($ref_file_copy);
    }
    else {
        vputs(">=", 2, "FAILED check against '$ref_file'!");

        if ( $testsuite::VERBOSITY >= 3
	     || $blackbox::OPT_INTERACTIVE ) {
            do_detailed_comparison($test_file, $ref_file, $ref_file_copy);
        }
    }

    return $code;
}

#
# blackbox::run_one_test
#
#   tests a set of command-line options, and run regression tests
#   on all the output files specified for that option set
#
#   arguments:
#
#     test_name
#       a string describing the test; by convention different
#       components of this name should be separated by '--'s
#
#     cmdline_args
#       a list of command-line arguments that should be passed
#        to the executable programming when running this test
#
#     files_to_check
#
#       * a list of names of output files that will be created by
#         running the executable with the given command-line options
#       * each file will be checked against known reference files
#       * default reference filename is
#         ${REF_DIR}/${test_name}--${file_name}
#       * to override the default reference filename, use a
#         two-element list in place of the output filename, with
#         the first element being the output filename, and the
#         second element being the name of the reference file to
#         be used for comparison
#
#         --> normally the reference file is taken to be
#             ${REF_DIR}/${test_name}--${new_name}
#         --> BUT, if this new name ends in a '/', then it is
#             taken to specify an alternate directory where the
#             ref file should be found, so the ref file is
#             ${new_name}/${test_name}--${file_name}
#
#     sharerefs (optional)
#       if present, this is a string name of another test whose
#       reference files we should share; so instead of looking up
#       reference files as ${REF_DIR}/${test_name}--{$file_name}, we
#       will look for ${REF_DIR}/${sharerefs}--${file_name}
#
# example:
#
#   blackbox::run_one_test
#       ("cool-output--1",
#        ['-fancyopt', '-cool-output', 'someinput.ppm'],
#        ['somelog.txt', 'img.ppm',
#         ['cool.ppm', 'reallycool.ppm']]);
#
# in this example, the executable will be called as "prog -fancyopt
# -cool-output someinput.ppm"
#
# then the following file comparisons will be made:
#   somelog.txt <--> ${REF_DIR}/cool-output--1--somelog.txt
#   img.ppm     <--> ${REF_DIR}/cool-output--1--img.ppm
#   cool.ppm    <--> ${REF_DIR}/cool-output--1--reallycool.ppm

# usage: run_one_test($test_name, $cmdline_opts, $files_to_check)
sub run_one_test {

    my $test_name = $_[0];
    my $cmdline_opts = $_[1];
    my $files_to_check = $_[2];
    my $sharerefs = $_[3];

    my $test_name_for_ref_files =
	(defined $sharerefs) ? $sharerefs : $test_name;

    # if we're pattern-matching on test names, then see if the
    # current test matches, and go on to the next test otherwise
    if ($test_name !~ m($blackbox::OPT_PATTERN)) {
        return @testsuite::TEST_RESULT_SKIPPED;
    }

    # if we're just listing test names, then just do that
    if ( $blackbox::OPT_LIST_TESTS != 0 ) {
        print $test_name . "\n";
        return @testsuite::TEST_RESULT_SKIPPED;
    }

    if ( $blackbox::OPT_LIST_REFS != 0 ) {

        foreach my $file (@$files_to_check) {

            my ($test_file, $ref_file) =
		get_test_and_ref_names($test_name_for_ref_files, $file);

            if (-f $ref_file) {
                print "$ref_file\n";
            }
	    elsif (-f "${ref_file}.gz") {
                print "${ref_file}.gz\n";
            }
	    else {
                die "\nOops! No such reference file: $ref_file\n";
            }
        }

        return @testsuite::TEST_RESULT_SKIPPED;
    }

    vputs(">=", 2, "\n=========================================================");
    vputs(">=", 1, "\ntest '$test_name' ... ");

    my @cmd = @$cmdline_opts;

    unshift @cmd, $blackbox::executable;
    if ($blackbox::EXTRA_ARGS ne "") {
	push @cmd, $blackbox::EXTRA_ARGS;
    }

    vputs(">=", 2, "\n\nrunning command '" . join("\n\t", @cmd) . "'\n");

    if (!open(CMDCHAN, join(" ", @cmd) . " 2>&1 |")) {
	vputs(">=", -1, "\nCouldn't run $blackbox::executable: $!\n");
	return @testsuite::TEST_RESULT_FAIL_AND_QUIT_NOW;
    }

    my @cmd_output = ();

    while (<CMDCHAN>) {
	push(@cmd_output, $_);

	if ($testsuite::VERBOSITY >= 4) {
	    print $_;
	}
    }

    close(CMDCHAN);
    my $code = $?;

    my $any_errors = 0;

    my $fatal_errors = 0;

    if ($code) {

        # If we are here, then the actual test command failed
        # (i.e. it crashed or something)... this is even a more
        # serious problem than if the command "succeeds" but
        # produces faulty output.
        $any_errors = 1;

	$fatal_errors = 1;

    } else {

        foreach my $file (@$files_to_check) {

            my ($test_file, $ref_file) =
		get_test_and_ref_names($test_name_for_ref_files, $file);

            my $res = check_one_file($test_file, $ref_file);

            if ($res != 0) {
                $any_errors = 1;
            }
        }
    }

    if ($any_errors) {
        vputs("<=", 1, "FAILED!");

        # For diagnostic purposes, we dump the command's stdout and
        # stderr here, if we didn't already print it (for high
        # verbosity) while the command was running.

        if ( $testsuite::VERBOSITY < 4 ) {
            vputs(">=", 2, "\n" . join("", @cmd_output));
        }
        vputs(">=", 2, "\ntest FAILED (command exited with exit status '$code'):");
    }
    else {
        vputs("==", 1, "ok");
    }

    vputs(">=", 2, "\n---------------------------------------------------------\n");

    if ($any_errors) {
        push @testsuite::FAILURES, $test_name;

	if ($fatal_errors) {
	    return @testsuite::TEST_RESULT_FAIL_AND_QUIT_NOW;
	}
	else {
	    return @testsuite::TEST_RESULT_FAIL;
	}
    }
    else {
        return @testsuite::TEST_RESULT_OK;
    }
}

# usage: run($target_program, @test_descrs)
sub run {
    my $target_program = shift;
    my @test_descrs = @_;

    $blackbox::executable = $target_program;

    testsuite::parse_options(\%blackbox::options, @ARGV);

    if ($blackbox::OPT_LIST_TESTS != 0) {
        $testsuite::VERBOSITY = -1;
    }
    if ($blackbox::OPT_LIST_REFS != 0) {
        $testsuite::VERBOSITY = -1;
    }

    my @test_scripts = ();

    foreach my $test (@test_descrs) {
	if (not defined $test->{name}) {
	    die "missing 'test' field in test descriptor\n";
	}
	if (not defined $test->{args}) {
	    die "missing 'args' field in test descriptor for $test->{name}\n";
	}
	if (not defined $test->{files}) {
	    die "missing 'files' field in test descriptor for $test->{name}\n";
	}

	push(@test_scripts,
	     sub { blackbox::run_one_test
		       ($test->{name},
			$test->{args},
			$test->{files},
			$test->{sharerefs}) });
    }

    # create a sandbox directory where we will run the tests:
    mkdir($blackbox::RUN_DIR) || die "Cannot mkdir($blackbox::RUN_DIR): ";

    # now push a cleanup handler to remove the sandbox directory that
    # will get run in our END block (but do this only if the user
    # hasn't requested --nocleanup):
    if ($blackbox::OPT_CLEANUP) {
	push (@blackbox::cleanup_code,
	      sub { chdir($blackbox::ROOT_DIR);
		    system("/bin/rm -rf $blackbox::RUN_DIR\n"); });
    }

    chdir($blackbox::RUN_DIR);

    # to allow testing of the --io=XXX stream input/output method, we
    # do need to have some test data in the sandbox directory. For
    # now, this is only going to be testpic001.pnm:
    system("/bin/ln -s ../inputs/testpic001.pnm testpic001.pnm");

    # run the tests in the sandbox:
    my $code = testsuite::run_tests($target_program, '../', @test_scripts);

    exit $code;
}

END {
    foreach my $code (@blackbox::cleanup_code) {
	&$code();
    }
}

1;
