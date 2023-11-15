#!/usr/bin/perl -w

# $Id: testlong_script_movies_blackbox.pl 10337 2008-10-13 17:24:09Z itti $

use strict;
use invt_config;

# Run a subset (--match) of the black box tests for the "ezvision"
# executable, except that we'll use the --executable option to
# substitute the bin/invt interpreter running the ezvision.tcl script
# instead:

my @args = ();

push @args, "$invt_config::abs_srcdir/testlong_movies_blackbox.pl";
push @args, "--match";
push @args, "movie-saccade--[14]";
push @args, "--executable";
push @args, "$invt_config::abs_top_srcdir/scripts/process_movie.pl "
    . "--rawframes --testmode --notmpdir "
    . "\"--vision-exec=$invt_config::exec_prefix/bin/invt -nw "
    . "$invt_config::abs_top_srcdir/scripts/ezvision.tcl\"";
push @args, @ARGV;

my $uname = `uname`;

if ($uname !~ /CYGWIN/) {

    exit system(@args);
}
