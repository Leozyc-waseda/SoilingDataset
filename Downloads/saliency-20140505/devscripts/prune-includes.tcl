#!/usr/bin/tclsh

# $Id: prune-includes.tcl 7882 2007-02-09 03:16:21Z itti $

# See usage examples in the makefile targets 'prune-source-includes'
# and 'prune-header-includes'.

# See bottom for main function, command-line options etc.

set WRITE_CHANGES 0
set CUMULATIVE 0
set DEFS { -DHAVE_CONFIG_H }
set CPPFLAGS { -I/lab/local/matlab/extern/include -I/lab/local/matlab/simulink/include -I/usr/local/xerces-c-1.7.0/include  -I/usr/X11R6/include -I/usr/lib/qt-3.3/include -fpic -I/usr/java/jdk1.5.0/include -I/usr/java/jdk1.5.0/include/linux  -I. -I/usr/include/xerces -I.ui -Isrc/qt -Isrc -include config.h }
if { [info exists ::env(CPPFLAGS)] } {
    set CPPFLAGS "$::env(CPPFLAGS) $CPPFLAGS"
}
set CXXFLAGS { -ansi -Wall -O3 -march=i686 }
set TRY_SYS_INCLUDES 1

proc write_all_lines_except { exclude_me lines fname } {
    set fd [open $fname "w"]
    for { set c 0 } { $c < [llength $lines] } { incr c } {
	if { $c != $exclude_me } {
	    puts $fd [lindex $lines $c]
	}
    }
    close $fd
}

proc resolve_include { fname } {
    if { [file exists src/$fname] } {
	return src/$fname
    }
    return ""
}

proc try_compile { origfname tmpfile what context } {

    set compileout "tmp-output-prune-includes-[pid]"

    set code [catch {eval exec g++ -x c++ $::DEFS $::CPPFLAGS \
			 $::CXXFLAGS -c $tmpfile -o /dev/null \
			 >& $compileout} result]

    set status [lindex "ok FAILED" $code]

    set errmsg ""

    set fd [open $compileout "r"]
    if { $code == 1 } {
	while { [gets $fd line] >= 0 } {
	    if { [regexp {^(.*):([0-9]+): error: (.*)} $line \
		      - srcname lineno errmsg] } {
		if { [string equal $srcname $tmpfile] } {
		    set errmsg " (line ${lineno}: $errmsg)"
		} else {
		    set errmsg " (${srcname}:line ${lineno}: $errmsg)"
		}
		break
	    }
	}
    }
    close $fd

    file delete $compileout

    puts [format "\[%s\]  %6s %s%s%s" \
	      $origfname $status $what $errmsg $context]

    return $code
}

proc prune_includes { fname } {

    set ::VISITED($fname) 1

    set fd [open $fname "r"]
    set lines [split [read $fd] "\n"]
    close $fd

    if { [string length [lindex $lines end]] == 0 } {
	set lines [lrange $lines 0 end-1]
    }

    set dir [file dirname $fname]
    set tail [file tail $fname]

    set tmpfile "${dir}/prune-includes-tmp-$tail"

    set context ""

    set anychanges 0

    if { [try_compile $fname $fname "as is" ""] != 0 } {
	puts "\[$fname\] WARNING $fname is already broken!"
	return
    }

    if { $::TRY_SYS_INCLUDES } {
	set pattern  {^\#include +(\"|<)([^\">]+)(\"|>)}
    } else {
	set pattern  {^\#include +(\")([^\">]+)(\")}
    }

    for { set c 0 } { $c < [llength $lines] } { incr c } {
	set line [lindex $lines $c]

	if { [regexp $pattern $line \
		 - lbracket incname rbracket] } {


	    if { [regexp {\.I$} $incname] } {

		# don't try to mess with .I files
		continue
	    } elseif { [string equal \
			    [file tail [file rootname $incname]] \
			    [file tail [file rootname $fname]]] } {

		# if we're analyzing baz.C, don't mess with baz.H
		continue
	    } elseif { [regexp {\.C$} $incname] } {

		# don't mess with .C files -- if someone is
		# #includ'ing a .C file, it probably means they need
		# some template instantiations from that file
		continue
	    }

	    if { 0 } {

		set resolved [resolve_include $incname]

		# don't do this recursion since it doesn't really work
		# as intended with header files -- some header files
		# require internal #includes for correctness, even
		# though they won't always generate an error if the
		# #include is missing (e.g., if an inline template
		# function uses assert(), the compiler won't complain
		# about <assert> being missing unless somebody
		# actually tries to call that inline template function).

		if { [string length $resolved] > 0 } {
		    if { ![info exists ::VISITED($resolved)] } {
			puts "\[$fname\]  recursing to visit $resolved"
			prune_includes $resolved
		    }
		}
	    }

	    write_all_lines_except $c $lines $tmpfile

	    set what "without $line"

	    set code [try_compile $fname $tmpfile $what $context]

	    file delete $tmpfile

	    if { $code == 0 && $::CUMULATIVE } {
		set anychanges 1
		set lines [lreplace $lines $c $c]
		append context [format "\n\[%s\]\t\tAND without %s" \
				    $fname $line]
		incr c -1 ;# we just erased a line, so back the counter up by 1
	    }
	}
    }

    if { $anychanges && $::WRITE_CHANGES } {
	write_all_lines_except -1 $lines $fname
	puts "\[$fname\]  NOTE: $fname modified!"
    } else {
	puts "\[$fname\]  $fname unmodified"
    }
}

##
## main code
##

if { [llength $argv] < 1 } {
    puts "USAGE: $argv0 ?options? <sourcefile>"
    exit 1
}

set fnames [list]

for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    switch -exact -- $arg {
	-DEFS {
	    incr i
	    set DEFS [lindex $argv $i]
	    puts "\[$argv0\]        DEFS is $DEFS"
	}
	-CPPFLAGS {
	    incr i
	    set CPPFLAGS [lindex $argv $i]
	    puts "\[$argv0\]        CPPFLAGS is $CPPFLAGS"
	}
	-CXXFLAGS {
	    incr i
	    set CXXFLAGS [lindex $argv $i]
	    puts "\[$argv0\]        CXXFLAGS is $CXXFLAGS"
	}
	-c {
	    puts "\[$argv0\]        operating in cumulative mode"
	    set CUMULATIVE 1
	}
	-w {
	    puts "\[$argv0\]  NOTE: will write changes back to original file"
	    set WRITE_CHANGES 1
	    set CUMULATIVE 1
	}
	--no-sys-includes {
	    puts "\[$argv0\]        will not consider \#include <>'s"
	    set TRY_SYS_INCLUDES 0
	}
	default {
	    lappend fnames $arg
	    puts "\[$argv0\]        will test source file $arg"
	}
    }
}

foreach fname $fnames {
    puts "\n==========================================================="
    puts "\[$argv0\]  trying source file $fname"
    prune_includes $fname
}
