# This Perl file contains common source code for running a series of
# test scripts.

# $Id: testrun.pm 10004 2008-07-30 02:12:59Z icore $

package testrun;

require Exporter;

@ISA = qw(Exporter);

use strict;
use Time::HiRes;

use invt_config;
use testsuite;

use POSIX;  # for signal handlers

# usage: display_system_info()
sub display_system_info {
    print "###\n";
    print "### system info ...\n";
    print "###\n";
    print "\n";

    my $date=`date`;
    print "DATE:\n\t$date\n";

    my $uname;
    if (-x '/bin/uname')        { $uname=`/bin/uname -a`; }
    elsif (-x '/usr/bin/uname') { $uname=`/usr/bin/uname -a`; }
    else                        { $uname='unavailable'; }
    print "UNAME:\n\t$uname\n";

    my $cxx = $invt_config::CXX;
    my $cxxver=`$cxx -v 2>&1`;
    my $whichcxx=`which $cxx`;
    chomp $whichcxx;
    print "CXX INFO ($whichcxx):\n\t$cxxver\n";

    my $abs_top_srcdir=$invt_config::abs_top_srcdir;

    if (-d "${abs_top_srcdir}/.svn") {
	my $svnver=`svnversion $abs_top_srcdir`;
	print "SVNVERSION:\n\t$svnver\n";

	my $svnstat=`svn status $abs_top_srcdir`;
	print "SVN STATUS:\n$svnstat\n";
    }
}


# usage: (nsuccess, ntotal, summary) = run_one_script(scriptfile, arglist)
sub run_one_script {
    my $script_file = $_[0];
    my $arglist = $_[1];

    my @ret; $ret[0] = 0; $ret[1] = 0; $ret[2] = "";

    open(TESTPROG, "$script_file $arglist 2>&1 |")
	or die "couldn't run test script $script_file!";

    my $script_output="";

    print "####################\n";
    print "### STARTING test script '$script_file' ...\n";
    print "###\n";

    while (<TESTPROG>) {
	print $_;
	$script_output .= $_;
	if (/^([0-9]+) of ([0-9]+) tests succeeded/) {
	    my $nsuccess=$1;
	    my $ntotal=$2;
	    my $marker="";
	    if ($nsuccess != $ntotal) {
		$marker=" FAIL==>";
	    }

	    $ret[2] .= sprintf("%-8s %3d of %3d tests succeeded (%s)\n",
			       $marker, $nsuccess, $ntotal, $script_file);

	    $ret[0] += $nsuccess;
	    $ret[1] += $ntotal;
	}
    }

    close TESTPROG;

    if ($? != 0) {
	print "\ntest suite aborted\n";
	$ret[0] = -1; $ret[1] = -1;
	return @ret;
    }

    print "###\n";
    print "### FINISHED test script '$script_file'\n";
    print "####################\n";
    print "\n";

    return @ret;
}

# usage: run_matching_scripts(glob_pattern)
sub run_matching_scripts {

    # force autoflush, so that we get line-buffered output
    my $oldh = select(STDOUT);
    $| = 1;
    select($oldh);

    display_system_info();

    my $glob_pattern = $_[0];

    my @test_script_files = sort(glob($glob_pattern));

    my $NSUCCESS=0;
    my $NTOTAL=0;
    my $results="";

    my $arglist=join(' ', @ARGV);

    foreach my $script_file (@test_script_files) {
	my @ret = run_one_script($script_file, $arglist);

	if ($ret[0] == -1 || $ret[1] == -1) {
	    # some script was aborted
	    return 1;
	}

	$NSUCCESS += $ret[0];
	$NTOTAL += $ret[1];
	$results .= $ret[2];
    }

    my $nfail = $NTOTAL - $NSUCCESS;

    if ($nfail == 0) {
	print "\nSUMMARY: ALL TESTS PASSED ($NSUCCESS of $NTOTAL)\n";
    }
    else {
	print "\nSUMMARY: TESTS FAILED!!! ($nfail of $NTOTAL)\n";
    }

    print $results;

    if ($nfail > 0) {
	print "\nWARNING: $nfail TESTS FAILED!!!\n";
    }

    print `date`;

    return $nfail;
}

# usage: benchmark_scripts(glob_pattern, benchfile)
sub benchmark_scripts {

    # force autoflush, so that we get line-buffered output
    my $oldh = select(STDOUT); $| = 1; select($oldh);

    my $glob_pattern = $_[0];
    my $benchfile = $_[1];
    my @test_script_files = sort(glob($glob_pattern));
    my $arglist=join(' ', @ARGV);
    my @torun;
    my $niter = 10;

    open (BENCH, ">$benchfile") or die "Cannot write $benchfile: ";

    print BENCH "# TestSuite benchmark hints (for parallel test ordering)\n";
    print BENCH "# ". `uname -a`;
    print BENCH "# ". `date` ."#\n";

    # compile a list of all the individual tests to run:
    foreach my $script_file (@test_script_files) {
	# if it's a blackbox, run each of the available tests as a
	# thread; otherwise, run the whole whitebox in one thread:
	if ($script_file =~ m/blackbox/) {
	    # get a list of available tests in this blackbox:
	    open(TESTPROG, "$script_file --list 2>&1 |")
		or die "couldn't run test script $script_file!";
	    while(<TESTPROG>)
	    { chomp; push(@torun, "$script_file --match $_"); }
	    close(TESTPROG);
	} else {
	    # just run the whole whitebox:
	    push(@torun, "$script_file");
	}
    }

    # let's do it!
    my $nscripts = $#torun + 1;
    print "Starting benchmark of $nscripts tests for $niter iterations...\n";

    foreach my $script_file (@torun) {
	my $starttim = Time::HiRes::time();
	my $iter = $niter;
	while ($iter > 0) {
	    system("$script_file $arglist 2>&1 >/dev/null");

	    if ($? != 0) {
		# some script was aborted
		print STDERR "$script_file failed. Make sure all tests pass ".
		    "before you benchmark!\n";
		return 1;
	    }
	    $iter --;
	}
	my $etim = (Time::HiRes::time() - $starttim) / $niter;

	print BENCH sprintf("%f %s\n", $etim, $script_file);
    }

    close BENCH;
    return 0;
}

my $running = 0; # number of threads running

# usage: run_matching_scripts_parallel(glob_pattern, numproc, [benchfile])
sub run_matching_scripts_parallel {

    # force autoflush, so that we get line-buffered output
    my $oldh = select(STDOUT);
    $| = 1;
    select($oldh);

    #display_system_info();

    my $glob_pattern = $_[0];
    my $nthreads = $_[1];
    my $benchfile = $_[2];

    # expand each chunk of the glob pattern and concatenate (as
    # opposed to expanding everything -- the goal os that people may
    # want to first provide a glob for long tests, followed by a glob
    # for shorter tests. Running the long ones first usully yields
    # less wait at the end of the list):
    my @glob_pats = split(/\s+/, $glob_pattern);
    my @test_script_files;
    foreach my $g (@glob_pats) { push(@test_script_files, sort(glob($g))); }

    my $arglist=join(' ', @ARGV);

    my @torun;
    $running = 0; # number of threads running

    # compile a list of all the individual tests to run:
    foreach my $script_file (@test_script_files) {
	# if it's a blackbox, run each of the available tests as a
	# thread; otherwise, run the whole whitebox in one thread:
	if ($script_file =~ m/blackbox/)
	{
	    # get a list of available tests in this blackbox:
	    open(TESTPROG, "$script_file --list 2>&1 |")
		or die "couldn't run test script $script_file!";
	    while(<TESTPROG>)
	    { chomp; push(@torun, "$script_file --match $_"); }
	    close(TESTPROG);
	}
	else
	{
	    # just run the whole whitebox:
	    push(@torun, "$script_file");
	}
    }

    # parallelize the scripts and collect results. The code here is a
    # simplified ripoff of beoqueue.pl:
    my $njob = $#torun + 1; my $job = 0; my $ii = 0;

    # setup signal handlers for our parallel job executions:
    $SIG{USR1} = \&error; $SIG{USR2} = \&warning;

    print "TESTSUITE: Parallelizing $njob tests over $nthreads threads.\n";

    # use ordering hints from a benchmark file?
    if ($benchfile) {
	print "TESTSUITE: Using benchmark hints from $benchfile\n";
	my %bench;
	open (BFIL, $benchfile) or die "couln't open $benchfile ";
	while(<BFIL>) {
	    # format is: time scriptname args
	    chomp; my @x = split(/\s+/, $_, 2);
	    next if ($x[0] =~ m/#/);
	    $bench{$x[1]} = $x[0];
	}
	close BFIL;

	my @torun2;
	foreach my $tr (@torun) {
	    if (!defined($bench{$tr})) {
		print STDERR "WARNING: no benchmark data for: $tr\n";
		print STDERR "  Use 'make testbench' to update data.\n";
		$bench{$tr} = 0.1;
	    }
	    push(@torun2, sprintf("%012d %s", $bench{$tr}*1.0e8, $tr));
	}

	@torun = ( );
	foreach my $tr (reverse sort @torun2) {
	    my @x = split(/\s+/, $tr, 2);
	    push(@torun, $x[1]);
	}
    }

    print "\n";
    my @resfiles;

    # go over every script to run:
    foreach my $s (@torun) {
	if ($running >= $nthreads)
	{
	    # everybody is busy; let's wait for one of our child
	    # processes to exit:

	    my $pid = wait();
	    $running--;
	}

	$job ++; # used to display job number
	my $ss = $s; $ss =~ s|^\./||;
	print "START:   $ss [$job/$njob]...\n";

	my $resname = testsuite::tempfile_name();
	push(@resfiles, $resname);

	my $res = fork();
	if (not defined $res) { die "Cannot fork: $!\n"; }
	elsif ($res == 0) { # we are the child: run job & exit
	    local *TESTP; # make sure each child gets a different filehandle
	    open(TESTP, "$s $arglist 2>&1 |")
		or die "couldn't run test script $s!";

	    while (<TESTP>) {
		if (/^([0-9]+) of ([0-9]+) tests succeeded/) {
		    my $nsuccess = $1;
		    my $ntotal = $2;
		    my $marker = "PASS:";
		    if ($nsuccess != $ntotal) {	$marker="   FAIL:"; }

		    print sprintf("%-8s %d of %d tests succeeded for %s\n",
				  $marker, $nsuccess, $ntotal, $s);

		    # print our results into a private results file:

		    local *RESF;
		    open RESF, ">$resname" or die "Cannot open $resname: ";
		    print RESF "$nsuccess $ntotal\n";
		    close RESF;
		}
	    }

	    close TESTP;
	    exit 0;
	} else {
	    # we are the parent: just note that we started a child:

	    $running ++;     # one more active thread
	}
    }

    # wait for all children to complete:
    while (wait() > 0)
    {
	# empty loop
    }

    # show final info:
    my $NSUCCESS = 0;
    my $NTOTAL = 0;

    foreach my $resname (@resfiles) {
	local *RESF;
	open(RESF, $resname) or die "Cannot open $resname for reading";
	my $res = <RESF>;
	close RESF;
	unlink $resname;
	chomp $res;
	my @x = split(/\s+/, $res);
	$NSUCCESS += $x[0];
	$NTOTAL += $x[1];
    }

    my $nfail = $NTOTAL - $NSUCCESS;
    if ($nfail == 0)
    { print "\nSUMMARY: ALL TESTS PASSED ($NSUCCESS of $NTOTAL)\n"; }
    else { print "\nSUMMARY: TESTS FAILED!!! ($nfail of $NTOTAL)\n"; }
    return $nfail;
}

######################################################################
sub error {
    msg("FATAL ERROR recieved from child");
    die "STOPPED!\n";
}

######################################################################
sub warning {
    msg("WARNING received from child node");
}

######################################################################
sub msg { # string
    my $dat = `/bin/date +"%y%m%d-%H:%M:%S"`; chomp $dat;
    print STDERR "BEOQUEUE $dat - $_[0]\n";
}

1;
