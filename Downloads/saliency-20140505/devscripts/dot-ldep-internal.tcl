#!/usr/bin/tclsh

# $Id: dot-ldep-internal.tcl 13884 2010-09-04 09:14:21Z itti $

# Script use to transform the output of devscripts/cdeps
# --output-ldep-raw into a format that is usable the graph-drawing
# tool 'dot'. This particular script extracts only the dependencies
# within a particular src subdirectory, and illustrates dependencies
# on external directories with a single graph edge. See 'make
# ldep-internals'.

proc strip_src_pfx { dirname } {
    if { [regexp {^src/(.*)} $dirname - stripped] } {
	return $stripped
    }
    return $dirname
}

proc safe_graphname { str } {
    return [string map {- _} $str]
}

proc get_rankdir { dirname } {

    if { [file exists ${dirname}/README.dxy] } {
	set fd [open ${dirname}/README.dxy]
	while { [gets $fd line] >= 0 } {
	    if { [regexp {rankdir: (..)} $line - rankdir] } {
		puts stderr "got rankdir $rankdir for $dirname"
		close $fd
		return $rankdir
	    }
	}
	close $fd
    }

    # else...
    return "TB"
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

proc get_file_briefdoc { fname } {
    set result $fname

    set fd [open $fname "r"]
    while { [gets $fd line] >= 0 } {
	if { [regexp {@file +([^ ]+) +(.*)} $line - filename doc] } {
	    set result $doc

	    # if the line is an end-of-comment, then we're done
	    if { [regexp {\*/$} $line] } { break }

	    # continue reading until the end of the comment or until a
	    # blank line:
	    while { [gets $fd line] >= 0 } {
		# if it's an empty line, we're done
		if { [string length $line] == 0 } { break }

		# otherwise, append the current line to our result
		append result " $line"

		# if the line is an end-of-comment, then we're done
		if { [regexp {\*/$} $line] } { break }

		# if the line has a <br> or a <p>, then we're done
		if { [regexp {<br>} $line] } { break }
		if { [regexp {<p>} $line] } { break }
	    }
	    break
	}
    }
    close $fd

    set result [cleanup_briefdoc $result]

    # strip any trailing junk from our result
    if { [regexp {(.*)\*/$} $result - stripped] } {
	set result $stripped
    }

    if { [regexp {(.*)<br>} $result - stripped] } {
	set result $stripped
    }

    if { [regexp {(.*)<p>} $result - stripped] } {
	set result $stripped
    }

    return [string trim $result]
}

if { [llength $argv] != 2 } {
    puts stderr "usage: $argv dirpfx ?filename?"
    puts stderr "\tFilters raw output from 'devscripts/cdeps --output-ldep-raw'"
    puts stderr "\tinto a format readable by 'dot'. If no 'filename' is given,"
    puts stderr "\tthen the program reads from stdin."
    exit 1
}

set dirpfx [string trimright [lindex $argv 0] "/"]

set fname [lindex $argv 1]
set fd [open $fname "r"]

while { [gets $fd line] >= 0 } {

    set f1 [lindex $line 0]
    set f2 [lindex $line 1]

    if { [string match "${dirpfx}*" $f1] } {

	set s1 [strip_src_pfx $f1]
	set s2 [strip_src_pfx $f2]

	if { ![info exists ::FILEDOC($s1)] } {
	    set ::FILEDOC($s1) [get_file_briefdoc $f1]
	}

	if { [string match "${dirpfx}*" $f2] } {

	    set DEPS([list $s1 $s2]) 1

	    if { ![info exists ::FILEDOC($s2)] } {
		set ::FILEDOC($s2) [get_file_briefdoc $f2]
	    }

	} else {

	    set extdir [file dirname $s2]

	    if { ![info exists ::DIRDOC($extdir)] } {
		set ::DIRDOC($extdir) [get_dir_briefdoc src/$extdir]
	    }

	    set EXTDEPS([list $s1 $extdir]) 1

	    set EXTDIRS($extdir) 1
	}
    }
}

set out stdout

set graphname [safe_graphname [strip_src_pfx $dirpfx]]
set rankdir [get_rankdir $dirpfx]

set ext_bordercolor "royalblue3"
set ext_fillcolor   "lightskyblue1"
set ext_edgecolor   "gray60"

set int_bordercolor "saddlebrown"
set int_fillcolor   "sandybrown"
set int_edgecolor   "saddlebrown"

puts $out "digraph $graphname {"
puts $out [format {%sgraph [rankdir=%s, concentrate=true];} "\t" $rankdir]
puts $out [format {%snode [shape=box, \
			       height=0.25, \
			       fontsize=9, \
			       fontname="Helvetica-Bold", \
			       peripheries=1, \
			       color=%s, \
			       fillcolor=%s, \
			       fontcolor=black, \
			       style="bold,filled"];} \
	       "\t" $int_bordercolor $int_fillcolor]

foreach dep [array names DEPS] {
    set f1 [lindex $dep 0]
    set f2 [lindex $dep 1]
    puts $out [format {%s"%s" -> "%s" [color=%s];} \
		   "\t" $f1 $f2 $int_bordercolor]
}

foreach dir [array names EXTDIRS] {
    puts $out [format {%s"%s" [URL="%s.html", \
				   label="[ext] %s", \
				   peripheries=2, \
				   color=%s, \
				   fillcolor=%s, \
				   fontcolor=black, \
				   shape=box, \
				   peripheries=2, \
				   style="bold,filled", \
				   tooltip="%s"];} \
		   "\t" $dir $dir $dir $ext_bordercolor $ext_fillcolor $::DIRDOC($dir)]
}
puts -nonewline $out "\t{ rank = sink; "
foreach dir [array names EXTDIRS] {
    puts -nonewline "\"$dir\"; "
}
puts $out "}"
foreach fname [array names FILEDOC] {
    if { [regexp (/ui/|/moc/) $fname] } {
	puts $out "\t\"$fname\" \[URL=\"#\", tooltip=\"auto-generated from .ui file\", style=\"dashed, filled\", fillcolor=antiquewhite1, fontcolor=gray40\];"
    } elseif { [regexp {\.(h|H|hpp)} $fname] } {
	puts $out "\t\"$fname\" \[URL=\"#\", tooltip=\"$::FILEDOC($fname)\", style=\"filled\", fillcolor=burlywood1\];"
    } else {
	puts $out "\t\"$fname\" \[URL=\"#\", tooltip=\"$::FILEDOC($fname)\"\];"
    }
}
foreach dep [array names EXTDEPS] {
    set f1 [lindex $dep 0]
    set d2 [lindex $dep 1]
    puts $out "\t\"$f1\" -> \"$d2\" \[color=$ext_edgecolor, style=dashed\];"
}
puts $out "}"

close $fd
