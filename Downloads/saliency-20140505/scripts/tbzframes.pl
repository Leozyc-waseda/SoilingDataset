#!/usr/bin/perl

# USAGE: tbzframes.pl <movie1.tbz> ... <movieN.tbz>
# will print out the frame range for each movie as:
# movie.tbz first last

foreach $f (@ARGV) {
    @range = sort(`tar jtvf $f`);
    $start = 0;
    while($range[$start] !~ m/frame/) { $start ++; }
    $end = $#range;
    while($range[$end] !~ m/frame/) { $end --; }
    print "$f ".frameno($range[$start])." ".frameno($range[$end])."\n";
}

sub frameno { # txt
    my @a = split(/[e\.]/, $_[0]);
    return $a[$#a - 1];
}
