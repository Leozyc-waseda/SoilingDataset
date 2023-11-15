#!/usr/bin/perl -w

# Courtesy robert.nielsen@everest.com
# from http://www.graphviz.org/Misc/dot_from_pl.pl

# with local modifications for the iLab Neuromorphic Vision C++ Toolkit

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/dot_from_pl.pl $
# $Id: dot_from_pl.pl 7446 2006-11-15 19:59:50Z rjpeters $

use strict;
use Data::Dumper;
use File::Basename;

$Data::Dumper::Purity = 1;
$Data::Dumper::Terse  = 1;

my $thresh = 0.01;
my $docroot = undef;

for (my $i = 0; $i < $#ARGV; $i++)
{
    if ($ARGV[$i] eq "-t" || $ARGV[$i] eq "--thresh")
    {
	$i++;
	# user specifies a percentage; we convert it to a fraction:
	$thresh = $ARGV[$i] / 100.0;
    }
    elsif ($ARGV[$i] eq "--docroot")
    {
	$i++;
	$docroot = $ARGV[$i];
    }
}

undef $/;
my $in = eval <STDIN>;
my $call_graph = $$in{call_graph};
my $overall_time = $$in{overall_time};
my $max_self_time = $$in{max_self_time};

my %edges;
my %nodes;

call_graph: while (my ($self_idx, $self) = each(%{$call_graph}))
{
    if (!defined($$self{name}) ||
	!defined($$self{parent_entries}) ||
	!defined($$self{child_entries}))
    {
	next call_graph;
    }

    if ($$self{name} =~ m/CYCLE/)
    {
	next call_graph;
    }

    if (!defined($$self{total_time}) ||
	$$self{total_time} < ($thresh * $overall_time))
    {
	next call_graph;
    }

    $nodes{$self_idx} = $self;

    foreach my $line (@{$$self{parent_entries}})
    {
	my $parent_idx = $$line{index};
	my $parent = $$call_graph{$parent_idx};

	if (defined($$parent{total_time})
	    && $$parent{total_time} >= ($thresh * $overall_time))
	{
	    $nodes{$parent_idx} = $parent;

	    my $edge = "\"$parent_idx\" -> \"$self_idx\"";
	    $edges{$edge} = 1;
	}
    }

    foreach my $line (@{$$self{child_entries}})
    {
	my $child_idx = $$line{index};
	my $child = $$call_graph{$child_idx};

	if (defined($$child{total_time})
	    && $$child{total_time} >= ($thresh * $overall_time))
	{
	    $nodes{$child_idx} = $child;

	    my $edge = "\"$self_idx\" -> \"$child_idx\"";
	    $edges{$edge} = 1;
	}
    }
}

sub escape_angle_brackets
{
    my $s = $_[0];
    $s =~ s/</\\</g;
    $s =~ s/>/\\>/g;
    return $s;
}

sub color_for_percent
{
    my $percent = $_[0];
    my $max_value = $_[1];

    my $hue = 0.4 - 0.4 * $percent / $max_value;
    my $sat = 0.05 + 0.95 * $percent / $max_value;
    my $bright = 1.0;

    if ($hue < 0) { $hue = 0; }
    if ($sat > 1) { $sat = 1; }

    if ($percent / $max_value < 0.01)
    {
	$sat = 0;
    }

    return sprintf("%.3f %.3f %.3f", $hue, $sat, $bright);
}

sub label_for_node
{
    my $node = $_[0];

    if (defined($$node{source}))
    {
	return sprintf("{ %s | %s | "
		       . "{ %d# | %.2f%% self | %.2f%% total } }",
		       escape_angle_brackets($$node{sig}),
		       $$node{source},
		       $$node{called},
		       $$node{self_percent},
		       $$node{total_percent});
    }
    else
    {
	return sprintf("{ %s | { %d# | %.2f%% self | %.2f%% total } }",
		       escape_angle_brackets($$node{sig}),
		       $$node{called},
		       $$node{self_percent},
		       $$node{total_percent});
    }
}

sub url_for_node
{
    my $node = $_[0];

    my $source = $$node{source};

    if (defined($docroot) && defined($source) && $source =~ m/(.*):(\d+)/)
    {
	my $file = $1;
	my $line = $2;

	my $base = basename($file);
	$base =~ s/_/__/;
	$base =~ s/\./_8/;

	return sprintf("%s/%s-source.html#l%05d",
		       $docroot, $base, $line);
    }
    else
    {
	return "";
    }
}

print "digraph g\n";
print "{\n";
print "  graph [rankdir=TB, size=\"16,20\"];\n";
print "  graph [ratio=compress, mclimit=10, ranksep=0.6, nodesep=0.4];\n";
print "  node  [shape=record, fontsize=16, fontname=\"Helvetica-Bold\", style=\"bold,filled\"];\n";
print "  edge  [arrowsize=1.5];\n";
while (my ($edge, $undef) = each(%edges))
{
    print "  $edge;\n";
}
my $total_percent = 0.0;
while (my ($node_idx, $node) = each(%nodes))
{
    printf("  \"%s\" [fillcolor=\"%s\", label=\"%s\", "
	   . "URL=\"%s\", tooltip=\"%s\"];\n",
	   $node_idx,
	   color_for_percent($$node{self_time}, $max_self_time),
	   label_for_node($node),
	   url_for_node($node),
	   $$node{sig});

    $total_percent += $$node{self_percent};
}
printf("  account [label=\"{ this graph accounts for %.2f%% of total CPU time "
       . "| including all functions using \\>= %.2f%% of CPU time }\", "
       . "shape=record, fontsize=20]\n", $total_percent, $thresh * 100.0);
print "}\n";
