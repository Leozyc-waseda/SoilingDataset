#!/usr/bin/perl

# add period, ppd, etc to e-ceyeS files
$usage = "USAGE: addmetadata.pl <period> <ppd> <trash> ".
    "<file1.e-ceyeS> [... <fileN.e-ceyeS>]\n";

my $per = shift || die $usage;
my $ppd = shift || die $usage;
my $tra = shift; #|| die $usage; 

while(my $fil = shift) {
    print STDERR "Converting $fil ...\n";
    my $didper = 0; my $didppd = 0; my $didtra = 0;
    open IN, $fil || die "Cannot read $fil: ";
    open OU, ">$fil.$$" || die "Cannot write $fil.$$: ";
    while(my $line = <IN>) {
	if ($line =~ m/^\s*\#/) {
	    print OU $line;   # comment line
	} elsif ($line =~ m/period\s*=/) {
	    if ($didper == 0) { print OU "period = $per\n"; $didper = 1; }
	} elsif ($line =~ m/ppd\s*=/) {
	    if ($didppd == 0) { print OU "ppd = $ppd\n";  $didppd = 1; }
	} elsif ($line =~ m/trash\s*=/) {
	    if ($didtra == 0) { print OU "trash = $tra\n"; $didtra = 1; }
	} else {
	    # it's a data line, make sure we have written out all metadata:
	    if ($didper == 0) { print OU "period = $per\n"; $didper = 1; }
	    if ($didppd == 0) { print OU "ppd = $ppd\n";  $didppd = 1; }
	    if ($didtra == 0) { print OU "trash = $tra\n"; $didtra = 1; }

	    print OU $line;
	}
    }
    close IN; close OU;
    system("/bin/mv -f $fil.$$ $fil");
}
