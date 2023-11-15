#!/usr/bin/perl -w

# $Id: test_scriptvision_blackbox.pl 10337 2008-10-13 17:24:09Z itti $

use strict;
use invt_config;

# Run a subset of the same black box tests as for the "ezvision"
# executable, except that we'll use the --executable option to
# substitute the bin/invt interpreter running the ezvision.tcl script
# instead:

my $uname = `uname`;

my $exename =
    "$invt_config::exec_prefix/bin/invt -nw "
    . "$invt_config::abs_top_srcdir/scripts/ezvision.tcl";

if ($uname !~ /CYGWIN/) {

    exit system("$invt_config::abs_srcdir/test_ezvision_blackbox.pl",
		"--match", "ez-.*channels.*", # subset of tests
		"--executable", $exename, @ARGV);
}
