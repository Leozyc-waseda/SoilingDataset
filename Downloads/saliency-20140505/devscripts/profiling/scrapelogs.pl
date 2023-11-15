#!/usr/bin/perl -w

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/scrapelogs.pl $
# $Id: scrapelogs.pl 6938 2006-08-04 23:05:48Z rjpeters $

use strict;
use Data::Dumper;

my $datdir = ".";

if (@ARGV)
{
    $datdir = $ARGV[0];
}

$Data::Dumper::Purity = 1;
$Data::Dumper::Terse  = 1;

my @sizes = qw(5120x3840 3584x2688 2560x1920 1792x1344
	       1280x960 896x672 640x480 448x336
	       320x240 224x168 160x120 112x84);

my @datlines;
my @pardatlines;

foreach my $sfx ("-t0", "-t15")
{
    size: foreach my $size (@sizes)
    {
	my $usersecs = 0.0;
	my $syssecs = 0.0;
	my $cpusecs = 0.0;
	my $realsecs = 0.0;
	my $MEMORY = 0;
	my $image_alloc = 0;
	my $block_size = 0;
	my $nblocks = 0;
	my $maxNormalizeFancy = 0;
	my $orientedPyramid = 0;
	my $motionPyramid = 0;
	my $lowPass = 0;
	my $evolve = 0;
	my $remainder = 0;

	my $logfile = "${datdir}/log${size}${sfx}.txt";
	open(FD, "<$logfile")
	    || die "${logfile}: $!\n";

	while (<FD>)
	{
	    if (/fastcache_alloc<64>:.*cache table entries in use, ([0-9\.]*)MB total allocated \(([0-9\.]*)kB \* *([0-9\.]*)\)/)
	    {
		$image_alloc = $1;
		$block_size = $2;
		$nblocks = $3;
	    }
	    elsif (/VmSize: *([0-9]*) kB/)
	    {
		$MEMORY = $1 / 1024.0;
	    }
	    elsif (/^([0-9\.]*)user ([0-9\.]*)system ([0-9]*):([0-9\.]*)elapsed/)
	    {
		$usersecs = $1;
		$syssecs = $2;
		$cpusecs = $1 + $2;
		$realsecs = 60 * $3 + $4;
	    }
	}
	close(FD);

	open(FD, "<${datdir}/gprof${size}${sfx}.pl")
	    || die "${datdir}/gprof${size}${sfx}.pl: $!\n";

	my $gprof;

	{
	    local $/;
	    undef $/;
	    $gprof = eval <FD>;
	}

	close(FD);

	my $maxremainder = 0;
	my $maxwhich = undef;

	while (my ($idx, $node) = each(%{$$gprof{call_graph}}))
	{
	    if ($$node{sig} =~ m/maxNormalize/
		|| $$node{sig} =~ m/sepFiltClean/
		|| $$node{sig} =~ m/[xy]FilterClean/)
	    {
		$maxNormalizeFancy += $$node{self_time};
	    }
	    elsif ($$node{sig} =~ m/orientedFilter/
		   || $$node{sig} =~ m/::operator-/
		   || $$node{sig} =~ m/quadEnergy/)
	    {
		$orientedPyramid += $$node{self_time};
	    }
	    elsif ($$node{sig} =~ m/ReichardtPyrBuilder/
		   || $$node{sig} =~ m/shiftImage/
		   || $$node{sig} =~ m/shiftClean/)
	    {
		$motionPyramid += $$node{self_time};
	    }
	    elsif ($$node{sig} =~ m/lowPass/)
	    {
		$lowPass += $$node{self_time};
	    }
	    elsif ($$node{sig} =~ m/[Ee]volve/
		   || $$node{sig} =~ m/getV/)
	    {
		$evolve += $$node{self_time};
	    }
	    else
	    {
		$remainder += $$node{self_time};
		if ($$node{self_percent} > $maxremainder)
		{
		    $maxremainder = $$node{self_percent};
		    $maxwhich = $$node{sig};
		}
	    }
	}

	printf("%4s %9s: mem = %7.2f MB, img mem = %7.2f MB, img = %6.1f kPix, B/Pix = %5.1f, "
	       . "cpu = %6.2f, real = %6.2f, kB/sec = %5.2f, maxNormalizeFancy = %f\n",
	       $sfx, $size, $MEMORY, $image_alloc, $block_size, $nblocks, $cpusecs, $realsecs,
	       $realsecs ? $block_size / $realsecs : 0, $maxNormalizeFancy);

	printf("max remainder in %s (%f)\n", $maxwhich, $maxremainder);

	my $dat =
	    #        1       2      3      4      5      6      7      8      9      10     11     12     13     14     15
	    sprintf("%7.2f  %7.2f  %6.1f  %5.1f  %6.2f  %4.2f  %6.2f  %6.2f  %5.2f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f",
		   $MEMORY, $image_alloc, $block_size, $nblocks, $usersecs, $syssecs, $cpusecs, $realsecs,
		   $realsecs ? $block_size / $realsecs : 0,
		    $maxNormalizeFancy, $orientedPyramid, $lowPass, $evolve, $motionPyramid, $remainder);

	if ($sfx eq "-t0") { push @datlines, $dat; }
	elsif ($sfx eq "-t15") { push @pardatlines, $dat; }
	else { die "bogus suffix $sfx\n"; }
    }
}

open(DAT, ">${datdir}/stats.txt")
    || die "${datdir}/stats.txt: $!\n";

while (@datlines and @pardatlines)
{
    my $dat = shift @datlines;
    my $pardat = shift @pardatlines;

    print(DAT "$dat  $pardat\n");
}

close(DAT);
