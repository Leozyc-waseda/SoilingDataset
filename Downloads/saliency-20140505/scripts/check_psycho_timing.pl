#!/usr/bin/perl

while(<STDIN>) {
    chomp; @x = split(/\s+/);
    if ($x[2] eq 'eyeTrackerCalibration') { $ecalib = $x[0]; } # begin
    if ($x[3] eq 'eyeTrackerCalibration') { # end
        print "eyeTrackerCalibration: ". timeDiff($ecalib, $x[0])."\n";
    }
    if ($x[2] eq 'displayFixationBlink') { $fixblink = $x[0]; } # begin
    if ($x[3] eq 'displayFixationBlink') { # end
        print "displayFixationBlink: ". timeDiff($fixblink, $x[0])."\n";
    }
}

sub timeDiff { # old, new
    my @ot = split(/[:\.]/, $_[0]); my @nt = split(/[:\.]/, $_[1]);
    my @td; my $n, $o, $b; my @base = qw/ 1000 60 60 1000 1000 /;
    while($n = pop(@nt)) {
        $o = pop(@ot); $b = pop(@base);
        my $diff = $n - $o;
        if ($diff < 0) { $diff += $b; $nt[$#nt] --; }
        push(@td, $diff);
    }
    return
        sprintf('%03d', $td[4]) . ':' .
        sprintf('%02d', $td[3]) . ':' .
        sprintf('%02d', $td[2]) . '.' .
        sprintf('%03d', $td[1]) . '.' .
        sprintf('%03d', $td[0]);
}
