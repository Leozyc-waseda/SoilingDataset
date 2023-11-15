#!/usr/bin/perl

# $Id: change-includes.pl 4724 2005-06-29 01:55:48Z rjpeters $

use strict;

use File::Basename;
use File::Path;

if (scalar(@ARGV) < 2) {
    die "usage: $0 old-include-name new-include-name ?file1 file2 ... ?\n";
}

my $oldinclude=$ARGV[0]; shift @ARGV;
my $newinclude=$ARGV[0]; shift @ARGV;

my $nchanged=0;

foreach my $fname (@ARGV) {

    my $fnewname = "./.devscripts-tmp/${fname}.new";
    my $fbkpname = "./.devscripts-tmp/${fname}.incbkp";

    my $tdir = dirname($fnewname);
    if (! -d $tdir) {
	if (mkpath($tdir) <= 0) {
	    die "Couldn't mkdir $tdir\n";
	}
    }

    $tdir = dirname($fbkpname);
    if (! -d $tdir) {
	if (mkpath($tdir) <= 0) {
	    die "Couldn't mkdir $tdir\n";
	}
    }

    open(SRCFILE, $fname)
	or die "Can't open $fname for reading\n";
    open(NEWSRCFILE, ">$fnewname")
	or die "Can't open $fnewname for writing\n";

    my $count = 0;

    while (my $line = <SRCFILE>) {
	$count += ($line =~ s/^\#( *)include "$oldinclude"/\#\1include "$newinclude"/);
	print NEWSRCFILE $line;
    }

    if ($count > 0) {
	print "$count change(s) in $fname ($oldinclude to $newinclude)\n";
	rename "$fname", "$fbkpname"
	    or die "Couldn't rename $fname to $fbkpname\n";
	rename "$fnewname", "$fname"
	    or die "Couldn't rename $fnewname to $fname\n";
	++$nchanged;
    }
    else {
	unlink "$fnewname" or die "Couldn't unlink $fnewname\n";
    }

    close NEWSRCFILE;
    close SRCFILE;
}

print "$nchanged file(s) changed ($oldinclude to $newinclude)\n";

exit
