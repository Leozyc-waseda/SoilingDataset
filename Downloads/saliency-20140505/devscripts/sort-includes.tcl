#!/usr/bin/tclsh

# $Id: sort-includes.tcl 4546 2005-06-15 02:27:31Z rjpeters $

# Quick script to alphabetically sort the #include directives within a
# file (or for each of a list of files).

# usage: ./devscripts/sort-includs.tcl file1 file2 file3 ...
#
# For each file named on the command line, the script alphabetically
# sorts consecutive #include lines. If there were any changes as a
# result of the sorting, then the script replaces the original file
# with the changed file. The script expects #include "foo.H", if
# present, to be the first #include in foo.C, and will complain
# otherwise. The script applies sorting only to consecutive #include
# lines, so #include groups that are separated by one or more empty
# lines will be sorted independently.

proc sort_includes { fname } {
    set fnewname "./.devscripts-tmp/${fname}.new"
    set fbkpname "./.devscripts-tmp/${fname}.sortbkp"
    file mkdir [file dirname $fnewname]
    file mkdir [file dirname $fbkpname]

    set fnamestem [file rootname [file tail $fname]]

    set ifd [open $fname "r"]
    set ofd [open $fnewname "w"]

    set nincludes 0

    set include_group [list]

    while { [gets $ifd line] >= 0 } {
	if { [regexp {^\#include (\"|<)([^\">]+)(\"|>)(.*)} $line \
		 - lbracket incname rbracket trailing] } {

	    incr nincludes

	    set incstem [file rootname [file tail $incname]]

	    set incext [file extension $incname]

	    # ok the current line is a #include; now, is it the .H
	    # file that matches a .C file?

	    if { [string equal $incext ".H"] && \
		     [string equal $fnamestem $incstem] } {
		if { $nincludes == 1 } {

		    # ok the first include is the matching .H file for
		    # a .C file, so let's write it immediately so it
		    # stays first
		    puts $ofd $line
		} else {
		    puts stderr "ERROR: in '$fname', '$incname' should be \#included first"
		    exit 1
		}
	    } else {

		# ok, the current line is a #include but is not the
		# matching .H file for a .C file, so let's save it in
		# the current include group, to be written when the
		# include group is over
		lappend include_group $line
	    }
	} else {

	    # ok, the current line is not a #include, so if we have
	    # any pending #include's, let's sort them and then write
	    # them

	    if { [llength $include_group] > 0 } {
		set include_group [lsort $include_group]
		foreach include $include_group {
		    puts $ofd $include
		}
		set include_group [list]
	    }

	    puts $ofd $line
	}
    }

    close $ofd
    close $ifd

    set code [catch {exec diff $fname $fnewname > /dev/null} result]

    if {$code == 0} {
	# no change after sorting
	file delete $fnewname
    } else {
	# file changed after sorting
	set wcc1 [lindex [exec wc -c $fname]    0]
	set wcc2 [lindex [exec wc -c $fnewname] 0]

	if { $wcc1 != $wcc2 } {
	    puts stderr "snafu in $::argv0: character count was changed during sorting"
	    puts stderr "original count was $wcc1, new count is $wcc2"
	    puts stderr "the original file '$fname' has been left untouched"
	    exit 1
	}

	puts "$fname changed after sorting"
	file mkdir [file dirname $fbkpname]
	file delete $fbkpname
	file rename $fname $fbkpname
	file rename $fnewname $fname
    }
}

foreach fname $argv {
    sort_includes $fname
}
