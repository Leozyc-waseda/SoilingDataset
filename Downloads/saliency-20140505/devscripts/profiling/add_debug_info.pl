#!/usr/bin/perl -w

# Filter to add source file and number information to a
# call-graph. Input is in .pl format (e.g. from pl_from_gprof.pl) and
# output is in an extended .pl format (e.g. suitable for input to
# dot_from_pl.pl).

# We get the source file information by calling gdb in batch mode;
# this means that you need to have the original executable around that
# generated the profile data in the first place.

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/add_debug_info.pl $
# $Id: add_debug_info.pl 7445 2006-11-15 19:40:16Z rjpeters $

use strict;
use Data::Dumper;

$Data::Dumper::Purity = 1;
$Data::Dumper::Terse  = 1;

if (scalar(@ARGV) != 1)
{
    print STDERR "usage: $0 exe-name  <in.pl  >out.pl\n";
    exit 1;
}

my $exename = $ARGV[0];

# Get the original call-graph:

my $in;
{
    local $/;
    undef $/;
    $in = eval <STDIN>;
}
my $call_graph = $$in{call_graph};

# Build a list of commands for gdb to get the addresses of all of the
# functions in the call-graph:

open(FD, ">cmds$$");
print FD "set print demangle off\n";

my @syms;
my @nodes;
while (my ($node_idx, $node) = each(%{$call_graph}))
{
    next if $$node{name} =~ m/CYCLE/;
    next if $$node{sig} =~ m/virtual thunk/;
    next if $$node{name} =~ m/__do_global_ctors_aux/;

    push @syms, $$node{name};
    push @nodes, $node;

    print FD "info address $$node{name}\n";
}

close(FD);

my @addrs;

open(FD, "-|", "gdb -x cmds$$ -batch $exename 2> /dev/null");
my $pos = 0;
while (<FD>)
{
    last if ($pos >= scalar(@syms));
    if ($_ =~ m/^Symbol "$syms[$pos]" is a function at address 0x([0-9a-f]+)\./)
    {
	push @addrs, $1;
	$pos++;
    }
    elsif ($_ =~ m/^Symbol "$syms[$pos]" is at 0x([0-9a-f]+) in a file compiled without debugging/)
    {
	push @addrs, $1;
	$pos++;
    }
    elsif ($_ =~ m/^Using host/)
    {
	# ignore
    }
    else
    {
	die "Don't know what to do with this:\n$_\n";
    }
}
close(FD);
unlink("cmds$$");

if (scalar(@addrs) != scalar(@syms))
{
    die "Oops! Didn't get an address for #$pos/$#syms $syms[$pos]\n";
}

# Now build a new list of commands to get the source file location of
# each of the function addresses:

open(FD, ">cmds$$");
foreach my $addr (@addrs)
{
    print FD "list *0x$addr\n";
}
close(FD);

open(FD, "-|", "gdb -x cmds$$ -batch $exename 2> /dev/null");
$pos = 0;
while (<FD>)
{
    last if ($pos >= scalar(@addrs));
    if ($_ =~ m/^0x$addrs[$pos] is in .* \((.*:\d+)\)\.$/m)
    {
	$nodes[$pos]{source} = $1;
	$pos++;
    }
}
close(FD);
unlink("cmds$$");

if ($pos != scalar(@addrs))
{
    die "Oops! Didn't get a location for #$pos $syms[$pos]\n";
}

print Dumper $in;
