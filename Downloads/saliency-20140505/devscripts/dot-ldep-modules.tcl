#!/usr/bin/tclsh

# $Id: dot-ldep-modules.tcl 5862 2005-11-03 20:29:35Z rjpeters $

# Script use to transform the output of devscripts/cdeps
# --output-ldep-raw into a format that is usable the graph-drawing
# tool 'dot'. See 'make ldep-modules.png'.

proc strip_subdirs { dirname } {

    if { [regexp {^([^/]+)/(.+)} $dirname - dir subdir] } {
	return $dir
    }

    return $dirname
}

proc strip_src_pfx { dirname } {
    if { [regexp {^src/(.*)} $dirname - stripped] } {
	return $stripped
    }
    return $dirname
}

# eliminate multiple spaces, and escape double-quotes so it's safe for
# a dot file
proc cleanup_briefdoc { str } {
    set str [regsub -all { +} $str { }]
    set str [string map {\" \\\"} $str]
    return $str
}

proc get_dir_briefdoc { dirname } {
    set result "$dirname directory"

    if { [file exists ${dirname}/README.dxy] } {

	set fd [open ${dirname}/README.dxy "r"]
	while { [gets $fd line] >= 0 } {
	    if { [regexp {\\brief (.*)} $line - doc] } {
		set result $doc
		break
	    }
	}
	close $fd
    }

    return [cleanup_briefdoc $result]
}

if { [llength $argv] == 0 } {
    set fd stdin
} elseif { [llength $argv] == 1 } {
    set fname [lindex $argv 0]
    set fd [open $fname "r"]
} else {
    puts stderr "usage: $argv ?filename?"
    puts stderr "\tFilters raw output from 'devscripts/cdeps --output-ldep-raw'"
    puts stderr "\tinto a format readable by 'dot'. If no 'filename' is given,"
    puts stderr "\tthen the program reads from stdin."
    exit 1
}

while { [gets $fd line] >= 0 } {

    set f1 [lindex $line 0]
    set f2 [lindex $line 1]

    set d1 [file dirname $f1]
    set d2 [file dirname $f2]

    set d1 [strip_src_pfx $d1]
    set d2 [strip_src_pfx $d2]

    set d1 [strip_subdirs $d1]
    set d2 [strip_subdirs $d2]

    if { ![string equal $d1 $d2] } {

	set DEPS([list $d1 $d2]) 1

	if { ![info exists ::DIRDOC($d1)] } {
	    set ::DIRDOC($d1) [get_dir_briefdoc src/$d1]
	}
	if { ![info exists ::DIRDOC($d2)] } {
	    set ::DIRDOC($d2) [get_dir_briefdoc src/$d2]
	}
    }
}

set out stdout

set bordercolor "royalblue3"
set fillcolor   "lightskyblue1"
set edgecolor   "gray25"

puts $out "digraph modules {"
puts $out "\tgraph \[rankdir=RL];"
puts $out "\tnode \[shape=box, height=0.25, fontname=\"Helvetica-Bold\", fontsize=10, peripheries=2, color=$bordercolor, fillcolor=$fillcolor, fontcolor=black, style=\"bold,filled\"\];"
foreach node [array names DIRDOC] {
    puts $out "\t\"$node\" \[URL=\"${node}.html\", tooltip=\"$::DIRDOC($node)\"\];"
}
foreach dep [array names DEPS] {
    set d1 [lindex $dep 0]
    set d2 [lindex $dep 1]
    puts $out "\t\"$d1\" -> \"$d2\" \[color=$edgecolor\];"
}
puts $out "}"

if { ![string equal $fd "stdin"] } {
    close $fd
}

