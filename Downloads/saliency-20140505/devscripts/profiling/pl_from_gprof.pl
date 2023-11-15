#!/usr/bin/perl -w

# Courtesy robert.nielsen@everest.com
# from http://www.graphviz.org/Misc/pl_from_gprof.pl

# with local modifications for the iLab Neuromorphic Vision C++ Toolkit

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/pl_from_gprof.pl $
# $Id: pl_from_gprof.pl 7445 2006-11-15 19:40:16Z rjpeters $

use strict;
use Data::Dumper;

$Data::Dumper::Purity = 1;
$Data::Dumper::Terse  = 1;

undef $/;
my $in = <STDIN>;
my ($flat, $call_graph_str, $index_str) = split /\cL/, $in;

$call_graph_str =~ s/^.*?index\s+%\s+time\s+self\s+children\s+called\s+name\n//s;

my @call_graph_entries = split /-+-\n/, $call_graph_str;

my %call_graph;

foreach my $lines (@call_graph_entries)
{
    my @lines = split /\n/, $lines;

    my $key = undef;

    my @parent_entries;
    my @child_entries;

    foreach my $line (@lines)
    {
	$line =~ s/<cycle (\d+) as a whole>/CYCLE$1/;
	$line =~ s/<cycle \d+>//;

	if ($line =~ /^\s*$/)
	{
	    # blank line; ignore
	}
	elsif ($line =~ m/^\[(\d+)\]\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*?)\s+\[(\d+)\]/)
	{
	    if (defined($key))
	    {
		die "too many keys in\n$lines!\n"
	    }

	    $key = $7;

	    $call_graph{$key}{percent_time} = $2;
	    $call_graph{$key}{self_time} = $3;
	    $call_graph{$key}{children_time} = $4;
	    $call_graph{$key}{called} = $5;
	    $call_graph{$key}{name} = $6;
	    $call_graph{$key}{total_time} = $3 + $4;

	    if ($call_graph{$key}{called} =~ m/^(\d+)\+(\d+)$/)
	    {
		# entries that are part of a cycle are listed with
		# calls=NNN+MMM, but we just want the NNN part:

		$call_graph{$key}{called} = $1;
	    }
	}
	elsif ($line =~ m/^\[(\d+)\]\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*?)\s+\[(\d+)\]/)
	{
	    if (defined($key))
	    {
		die "too many keys in\n$lines!\n"
	    }

	    $key = $6;

	    $call_graph{$key}{percent_time} = $2;
	    $call_graph{$key}{self_time} = $3;
	    $call_graph{$key}{children_time} = $4;
	    $call_graph{$key}{called} = "0";
	    $call_graph{$key}{name} = $5;
	    $call_graph{$key}{total_time} = $3 + $4;
	}
	elsif ($line =~ m/^\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*?)\s+\[(\d+)\]/)
	{
	    my $entry = { 'self' => "$1",
			  'children' => "$2",
			  'called' => "$3",
			  #'name' => "$4",
			  'index' => "$5" };

	    if (defined($key))
	    {
		push @child_entries, $entry;
	    }
	    else
	    {
		push @parent_entries, $entry;
	    }
	}
	elsif ($line =~ m/^\s+(\d+)\s+(.*?)\s+\[(\d+)\]/)
	{
	    # ignore
	}
	elsif ($line =~ m/^\s+<spontaneous>/)
	{
	    # ignore
	}
	else
	{
	    print STDERR "IGNORING: $line\n";
	}
    }

    if (!defined($key)) {
	die "no key in\n$lines\n";
    }

    $call_graph{$key}{parent_entries} = \@parent_entries;
    $call_graph{$key}{child_entries} = \@child_entries;
}


$flat =~ s/^\s*[Ff]lat\s+profile:\s+.*?\s+time\s+seconds\s+seconds\s+calls\s+[Kmu]?s\/call\s+[Kmu]?s\/call\s+name\s*\n//s;

my $overall_time = 0.0;

foreach my $line (split /\n/, $flat)
{
    my ($time, $cum_sec, $self_sec, $calls,
	$self_per_call, $total_per_call, $name);

    if ($line =~ m/\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)\s*/)
    {
	$time = $1;
	$cum_sec = $2;
	$self_sec = $3;
	$calls = $4;

	# these are unreliable because the units in the gprof file
	# might be s/call, or ms/call, or us/call, etc.:

	#$self_per_call = $5;
	#$total_per_call = $6;

	$name = $7;
    }
    elsif ($line =~ m/\s*(\S+)\s+(\S+)\s+(\S+)\s+(.*)\s*/)
    {
	$time = $1;
	$cum_sec = $2;
	$self_sec = $3;
	$calls = 0;

	$self_per_call = undef;
	$total_per_call = undef;

	$name = $4;
    }
    else
    {
	die "Bogus flat profile line:\n$line\n";
    }

    if ($cum_sec > $overall_time)
    {
	$overall_time = $cum_sec;
    }
}

sub cleanup_sig
{
    my $sig = $_[0];
    $sig = `c++filt "$sig"`;
    chop $sig;
    $sig =~ s/\(.*$//;
    $sig =~ s/promote_trait<float, float>::TP/float/;
    return $sig;
}

my $max_self_time = 0.0;

while (my ($key, $value) = each(%call_graph))
{
    $$value{self_percent} = 100.0 * $$value{self_time} / $overall_time;
    $$value{children_percent} = 100.0 * $$value{children_time} / $overall_time;
    $$value{total_percent} = 100.0 * $$value{total_time} / $overall_time;

    if ($$value{self_time} > $max_self_time)
    {
	$max_self_time = $$value{self_time};
    }

    $$value{sig} = cleanup_sig($$value{name});
}

my %root;
$root{overall_time} = $overall_time;
$root{max_self_time} = $max_self_time;
$root{call_graph} = \%call_graph;

print Dumper \%root;
