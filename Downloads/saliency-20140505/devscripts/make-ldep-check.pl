#!/usr/bin/perl -w

# $Id: make-ldep-check.pl 4812 2005-07-05 22:25:10Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/make-ldep-check.pl $

use strict;

if (scalar(@ARGV) != 1) {
    die "usage: $0  old-ldep-modules\n";
}

open (LDEPOLD, $ARGV[0])
    or die "Couldn't open $ARGV[0] for reading\n";

my %deps;

# build up a hash of hashes that lists the old ldeps
while (<LDEPOLD>) {
    my @words = split;
    $deps{$words[0]}{$words[1]} = 1;
}

close LDEPOLD;

my $exit_status = 0;

while (<STDIN>) {
    my @words = split;
    unless (defined $deps{$words[0]}{$words[1]}) {
	print "dependency not allowed in '$ARGV[0]': ";
	print "$words[0] --> $words[1]\n";
	$exit_status++;
    }
}

exit $exit_status;
