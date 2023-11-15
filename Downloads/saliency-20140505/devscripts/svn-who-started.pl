#!/usr/bin/perl -w

# Quick script to check who committed the first revision of a
# particular source file. Currently there are certainly many bugs and
# missing features. Example usage:
#
#   svn-who-started.pl src/Image/Image.H src/INVT/ezvision.C
#
# gives
#
#   src/Image/Image.H: itti@src2-r48 ( it's here -- template version in src2/)
#   src/INVT/ezvision.C: itti@src3-r2268 ( merging branch rev31 back into the trunk. Everything is broken (but test suite passes!). More info to come soon. Most problematic was Dirk's renaming of a bunch of files, so I am not sure how well my manual conflict resolution went on those. Remember, 'cvs co -r PRE31 saliency' will give you the last stable version.)

# $Id: svn-who-started.pl 7887 2007-02-09 19:01:28Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/svn-who-started.pl $

use strict;

use File::Basename;

# Hack for files that have undergone unusual name changes (FIXME: is
# there a way to get that information automatically from the svn log?)
my %hack = (
    'Beobot/test-BeobotControl.C' => 'test-carcontrol.C'
);

sub get_log_entry {
    my $fname = shift;

    open(SVNCHAN, "svn log $fname |")
	or die "svn log failed";

    my $rev = -1;
    my $user = "";
    my $entry = "";

    while (<SVNCHAN>) {
	chomp;
	if (/^r([0-9]+) \| ([a-z0-9]+) \| .* lines?$/) {
	    $rev = $1;
	    $user = $2;
	    $entry = "";
	}
	elsif (m/^--------/) {
	    next;
	}
	elsif (m/^$/) {
	    next;
	}
	else {
	    $entry .= " $_";
	}
    }

    close(SVNCHAN);

    return ($rev, $user, $entry);
}

foreach my $fullname (@ARGV) {

    if ($fullname !~ m{^src/(.*)}) {
	next;
    }

    my $filename = $1;

    my ($rev, $user, $entry) = get_log_entry("src/$filename");
    my $dir = "src3";

    if ($entry =~ m/[Bb]ranch.*src2/) {
	my $src2fname;
	if (exists($hack{$filename})) {
	    $src2fname = $hack{$filename};
	}
	else {
	    $src2fname = basename($filename);
	    if ($src2fname =~ m/^app-(.*)/) {
		$src2fname = $1;
	    }
	}
	($rev, $user, $entry) =
	    get_log_entry("svn://ilab.usc.edu/trunk/saliency/src2/$src2fname\@$rev");
	$dir = "src2";
    }

    die "no user for $filename" if $user eq "";
    die "no rev for $filename" if $rev < 0;

    printf("%s: %s\@%s-r%d (%s)\n", $fullname, $user, $dir, $rev, $entry);
}
