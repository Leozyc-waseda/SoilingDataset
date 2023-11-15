#!/usr/bin/tclsh

# $Id: make-gccfulldeps.tcl 6004 2005-11-29 17:23:21Z rjpeters $

# Builds a full list of compile dependencies, using g++ -M.

set CXX       $::env(CXX)
set DEFS      $::env(DEFS)
set CPPFLAGS  $::env(CPPFLAGS)
set SOURCES   $::env(SOURCES)

foreach f [lsort $SOURCES] {
    puts stderr $f
    set deps [eval exec $CXX $DEFS $CPPFLAGS -M -MG -x c++ $f]
    set deps [string map {"\\\n" " "} $deps]
    if { ![regexp {:(.*)} $deps - deplist] } {
        puts stderr "couldn't parse deps line '$deps'"
        exit 1
    }
    set deplist2 [list]
    foreach dep $deplist {
	set dep [file normalize $dep]
	if { [regexp "[pwd]/(.*)" $dep - reldep] } {
	    lappend deplist2 $reldep
	} else {
	    lappend deplist2 $dep
	}
    }
    foreach dep [lsort -unique $deplist2] {
        puts stdout [format "%-40s  %s" $f $dep]
    }
}
