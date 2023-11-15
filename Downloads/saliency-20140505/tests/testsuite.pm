# This Perl file contains routines that are shared amongst the various
# test scripts.

# $Id: testsuite.pm 8153 2007-03-21 00:04:26Z rjpeters $

package testsuite;

require Exporter;

@ISA = qw(Exporter);
@EXPORT = qw(vputs);

use strict;
use Cwd;

use invt_config;

# Each of these is basically a list/bitfield with three bits:
#   {was_test_run? did_test_pass? quit_now?}
@testsuite::TEST_RESULT_OK = (1, 1, 0);
@testsuite::TEST_RESULT_FAIL = (1, 0, 0);
@testsuite::TEST_RESULT_FAIL_AND_QUIT_NOW = (1, 0, 1);
@testsuite::TEST_RESULT_SKIPPED = (0, 0, 0);

$testsuite::VERBOSITY = 3;
$testsuite::TEMP_COUNTER = 0;
$testsuite::OPT_QUIT_ON_FAIL = 0;
$testsuite::EXIT_NOW = 0;
$testsuite::EXIT_REASON = "";
@testsuite::FAILURES = ();
@testsuite::TEMP_FILES = ();
$testsuite::DO_PROFILING = 1;
$testsuite::SAVE_GMONS = 0;
$testsuite::SHOW_HELP = 0;
$testsuite::OPT_DEBUG = 0;

# usage: vputs($op, $level, @args) e.g., vputs("<=", 2, args)
sub vputs {
    my $op = shift @_;
    my $level = shift @_;
    my @args = @_;

    my $doprint = 0;
    if ($op eq "==" && $testsuite::VERBOSITY == $level) {
	$doprint = 1;
    }
    elsif ($op eq "<=" && $testsuite::VERBOSITY <= $level) {
	$doprint = 1;
    }
    elsif ($op eq ">=" && $testsuite::VERBOSITY >= $level) {
	$doprint = 1;
    }

    if ($doprint) {
	foreach my $msg (@args) {
	    print STDERR $msg;
	}
    }
}

# usage: linewrap($text, $linelength)
sub linewrap {
    my $text = shift @_;
    my $linelength = shift @_;

    my @lines = ();

    my $line = "";

    foreach my $rawline (split("\n", $text)) {

	foreach my $word (split(/ +/, $rawline)) {
	    $word =~ s/\t/    /g;
	    if (length($line) == 0 ||
		length($line) + length($word) < $linelength) {
		$line .= "$word ";
	    }
	    else {
		push(@lines, $line);
		$line = "$word ";
	    }
	}

	push(@lines, $line);
	$line = "";
    }

    return @lines;
}

# Go through each of our possible command-line options, see if it is
# present on the command-line, and if so, get the option's argument
# and set the relevant variable to that value.

# usage: parse_options(@ARGV)
sub parse_options {
    my $optlist = shift;
    my @args = @_;

    for (my $i = 0; $i < scalar @args; $i++) {

	if (defined $$optlist{$args[$i]}) {

	    my $opt = $$optlist{$args[$i]};

	    if (defined $opt->{optval}) {

		# no argument required for this option
		${$opt->{varref}} = $opt->{optval};
	    }
	    elsif ($i+1 < scalar @args) {

		${$opt->{varref}} = $args[$i+1];
		if ($testsuite::OPT_DEBUG) {
		    vputs(">=", 2, "\ngot $args[$i] '$args[$i+1]'\n");
		}
	    }
	    else {

		# no default value, and no value given on the
		# command-line, so bail out:
		die "\nERROR: no value given for option '$args[$i]'\n";
	    }
	}
    }

    if ($testsuite::SHOW_HELP) {
        print STDERR "\nusage: $0 [OPTIONS]\n";
        foreach my $optname (sort keys %$optlist) {
	    my $opt = $$optlist{$optname};

	    my $msg = "  $optname";
	    if (not defined $opt->{optval}) {
		$msg .= " <arg>";
	    }
	    my $help = "";
	    if (defined $opt->{help}) {
		my @lines = linewrap($opt->{help}, 50);
		$help = join("\n" . (" " x 26), @lines);
		#$help = " $opt->{help}";
	    }

	    printf STDERR "%-25s %s\n", $msg, $help;
        }
        exit(1);
    }
}

# Convert any pathname into an absolute pathname, by prepending the
# current working directory if necessary.
# usage: absolute_path($path)
sub absolute_path {
    my $path = $_[0];
    if ($path =~ m|^/|) {
	return $path;
    }
    else {
	return cwd() . "/$path";
    }
}

# usage: tempfile_name()
sub tempfile_name {
    $testsuite::TEMP_COUNTER++;
    my $name = sprintf("test_suite_temp_%d_%d",
		       $$, $testsuite::TEMP_COUNTER);
    push @testsuite::TEMP_FILES, $name;
    return $name;
}

sub cleanup_temp_files {
    while (scalar @testsuite::TEMP_FILES > 0) {
	my $f = shift @testsuite::TEMP_FILES;
	$f =~ m/test_suite_temp_(\d+)_(\d+)/;

	# only remove the file it was created by us (and not our
	# parent process or some other process id):
	if ($1 == $$) {
	    unlink $f;
	}
    }
}

# return the entire contents of the named file
# usage: contents_of(filename)
sub contents_of {
    open CHAN, "< $_[0]";
    my @lines = <CHAN>;
    close CHAN;
    return join("", @lines);
}

# return a list of the lines in the named file
# usage: contents_of(filename)
sub lines_of {
    open CHAN, "< $_[0]";
    my @lines = <CHAN>;
    close CHAN;
    return @lines;
}

# Build a profile summary file out of a set of "gmon.out"-type
# files.
# usage: make_prof_summary($executable, $outdir, @gmons)
sub make_prof_summary {

    my $executable = shift;
    my $outdir = shift;
    my $gmons = join ' ', @_;

    if (length($gmons) == 0) { return; }

    my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) =
	localtime(time);

    my $timestamp = sprintf("%04d-%02d-%02d",
			    $year+1900,$mon+1,$mday);

    my $exestem = "unknown";

    if ($executable =~ m|([^/]+)$|) {
	$exestem = $1;
	if ($exestem =~ m|(.+)\.([^\.]+)$|) {
	    $exestem = $1;
	}
    }

    my $counter = 0;
    my $profile_file;
    do {
	$counter++;
	$profile_file = sprintf("%sprofile-%s-%s-%02d.txt",
				$outdir, $timestamp, $exestem, $counter);
    } while (-f $profile_file);

    my $result =
	system("gprof $executable $gmons > $profile_file");

    if ($result != 0) {
	die "error while running gprof: $?\n";
    }

    vputs(">=", 0,
	  "\nwrote $profile_file with summary of $gmons\n");
}

# We expect test_scripts to be a list, each item of which is a an
# anonymous perl "sub" that will return a a test result code. $outdir
# should point to a directory where we can save any non-transient
# results from running the test (such as profile summaries).

# usage: run_tests($executable, $outdir, @test_scripts)
sub run_tests {

    my $executable = shift;
    my $outdir = shift;
    my @test_scripts = @_;

    my $return_code = 0;

    if ($testsuite::VERBOSITY < -1 ||
	$testsuite::VERBOSITY > 4) {
	die "\ninvalid --verbosity ($testsuite::VERBOSITY); must be between -1 and 4, inclusive";
    }

    # If we're quitting when a test fails, then change the
    # relevant bit in TEST_RESULT_FAIL.
    if ($testsuite::OPT_QUIT_ON_FAIL) {
        @testsuite::TEST_RESULT_FAIL =
	    @testsuite::TEST_RESULT_FAIL_AND_QUIT_NOW;
    }

    # clear any previous profiling files before we start this
    # batch of tests
    if (-f "gmon.out") { unlink "gmon.out"; }

    my @gmons = ();

    my $num_tests_run = 0;
    my $num_tests_ok = 0;

    foreach my $script (@test_scripts) {

	if ($testsuite::EXIT_NOW) {
	    print STDERR "$testsuite::EXIT_REASON\n";
	    last;
	}

        my @result = &$script();

        # if this test run generated a gmon.out file with
        # profiling data, then move it aside to a unique filename
        # for later use
        if (-f "gmon.out") {
            my $profcounter = 1;
            while (-f "gmon.out$profcounter") {
                $profcounter++;
            }
            rename("gmon.out", "gmon.out$profcounter");
	    push @gmons, "gmon.out$profcounter";
        }

        $num_tests_run += $result[0];
        $num_tests_ok  += $result[1];

        if ( $result[2] ) {
            vputs(">=", 0,
		  "\nNOTE:\tSome tests may have been skipped, because either the option "
		  . "\n\t'--quit-on-fail' was set, or because the executable "
		  . "\n\tprogram itself failed. For more detailed "
		  . "\n\tresults, re-run the suite without '--quit-on-fail' "
		  . "\n\tor with a higher '--verbosity' setting.");
            my $return_code = -1;
	    last;
        }

	testsuite::cleanup_temp_files();
    }

    vputs(">=", 0, "\n$num_tests_ok of $num_tests_run tests succeeded\n");

    # if there was an associated executable file, and we generated
    # any raw profile-out files in this test run (i.e. gmon.out*
    # files), then compile them together into a profile-summary
    # file

    if ($testsuite::DO_PROFILING && length($executable)) {
	testsuite::make_prof_summary($executable, $outdir, @gmons);

    }

    if ($testsuite::SAVE_GMONS == 0) {
	foreach my $f (@gmons) { unlink $f; }
    }

    if (scalar @testsuite::FAILURES > 0) {
        vputs(">=", 0, "\nFAILED tests:\n");
        foreach my $f (@testsuite::FAILURES) {
	    vputs(">=", 0, "\t$f\n");
	}
    }

    return $return_code;
}

END {
    testsuite::cleanup_temp_files();
}

sub sighandler {
    my $signame = shift;
    $testsuite::EXIT_REASON = "caught SIG$signame";
    $testsuite::EXIT_NOW = 1;
}

$SIG{INT} = \&testsuite::sighandler;

1;
