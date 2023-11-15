#!/usr/bin/tclsh

# $Id: newclass.tcl 13100 2010-03-30 21:38:50Z itti $

proc show_usage {} {

    puts stderr "usage: newclass.tcl ?options...? <FileStem1> \[<FileStem2> \[<FileStem3> ...\]\]"
    puts stderr "       Will create FileStem.H and FileStem.C with contents as follows."
    puts stderr "       In each file it places the following:"
    puts stderr ""
    puts stderr "       (1) opening doxygen file comment (/*!@file ... */)"
    puts stderr "       (2) the contents of the devscripts/COPYRIGHT file"
    puts stderr "       (3) maintainer information and the svn Id string"
    puts stderr "       (4) start of the include guard (\#ifndef/\#define)"
    puts stderr "       (5) (.C file only) \#include the corresponding .H file in the .C file"
    puts stderr "       (6) emacs boilerplate (indent-tabs-mode: nil)"
    puts stderr "       (7) end of the include guard (\#endif)"
    puts stderr ""
    puts stderr "options:"
    puts stderr "-help  show this help message"
    puts stderr "-qt    generated files will have .H/.Q extensions instead of .H/.C"
    puts stderr "-c     generated files will have .h/.c extensions instead of .H/.C"
}

proc internal_name { fname } {

    # strip a leading src/ prefix, if it exists
    if { [regexp {^src/(.*)} $fname - subname] } {
        return $subname
    }

    return $fname
}

proc write_file_comment { fd fname } {
    puts $fd [format {/*!@file %s */} $fname]
    puts $fd ""
}

proc write_file_contents { fd fname } {
    set ifd [open $fname "r"]
    while { [gets $ifd line] >= 0 } {
        puts $fd $line
    }
    close $ifd
}

proc write_maintainer { fd } {
    set maintainer ""

    if { [file exists $::env(HOME)/.invt] } {
        set ff [open $::env(HOME)/.invt r]
        while { [gets $ff line] >= 0 } {
            if { [regexp {^maintainer:(.*)} $line - m] } {
                set maintainer " [string trim $m]"
                break
            }
        }
        close $ff
    }

    puts $fd "//"
    puts $fd "// Primary maintainer for this file:$maintainer"
    puts $fd [format {// $%s$} "HeadURL"]
    puts $fd [format {// $%s$} "Id"]
    puts $fd "//"
    puts $fd ""
}

proc include_guard_name { fname } {

    # change invalid characters for c++ identifiers (e.g. '.', '-')
    # into underscores, then change the whole thing to uppercase:
    set nm [string toupper [string map ". _ - _ / _" $fname]]
    return ${nm}_DEFINED
}

proc write_include_guard_start { fd fname } {
    set g [include_guard_name $fname]
    puts $fd [format "\#ifndef %s" $g]
    puts $fd [format "\#define %s" $g]
    puts $fd ""
}

proc write_include_guard_end { fd fname } {
    set g [include_guard_name $fname]
    puts $fd [format "\#endif // %s" $g]
}

proc write_emacs_boilerplate { fd lang } {
    puts $fd [format {// %s} [string repeat "\#" 70]]
    puts $fd {/* So things look consistent in everyone's emacs... */}

    # This line is obfuscated just a bit so that emacs doesn't try to
    # put THIS file (i.e. this tcl script) in c++ mode:

    puts $fd [format {/* %s %s: */} "Local" "Variables"]

    switch -exact -- $lang {
	c {
	    puts $fd {/* indent-tabs-mode: nil */}
	    puts $fd {/* c-file-style: "linux" */}
	}

	c++ {
	    puts $fd {/* mode: c++ */}
	    puts $fd {/* indent-tabs-mode: nil */}
	}

	default {
	    error "invalid language: '$lang'"
	}
    }

    puts $fd {/* End: */}
    puts $fd ""
}

proc write_include_header { fd fname } {
    puts $fd [format "\#include \"%s\"" $fname]
    puts $fd ""
}

proc newclass { filestem hext cext lang } {

    set copyright_file [file dirname [info script]]/COPYRIGHT

    if { [file exists ${filestem}.${hext}] } {

        puts stderr "NOTE: ${filestem}.${hext} already exists and has been left untouched"

    } else {

        set fd [open ${filestem}.${hext} "w"]
        write_file_comment        $fd [internal_name ${filestem}.${hext}]
        write_file_contents       $fd $copyright_file
        write_maintainer          $fd
        write_include_guard_start $fd [internal_name ${filestem}.${hext}]
        write_emacs_boilerplate   $fd $lang
        write_include_guard_end   $fd [internal_name ${filestem}.${hext}]
        close $fd

        puts "generated ${filestem}.${hext}"
    }

    if { [file exists ${filestem}.${cext}] } {

        puts stderr "NOTE: ${filestem}.${cext} already exists and has been left untouched"

    } else {
        set fd [open ${filestem}.${cext} "w"]
        write_file_comment        $fd [internal_name ${filestem}.${cext}]
        write_file_contents       $fd $copyright_file
        write_maintainer          $fd
        # write_include_guard_start $fd [internal_name ${filestem}.${cext}]
        write_include_header      $fd [internal_name ${filestem}.${hext}]
        write_emacs_boilerplate   $fd $lang
        # write_include_guard_end   $fd [internal_name ${filestem}.${cext}]
        close $fd

        puts "generated ${filestem}.${cext}"
    }
}

if { [llength $argv] == 0 } {
    show_usage

    exit 1
}

set hext "H"
set cext "C"
set lang "c++"

foreach arg $argv {
    switch -glob -- $arg {
        -qt {
            set hext "H"
	    set cext "Q"
	    set lang "c++"
        }
        -c {
	    set hext "h"
	    set cext "c"
	    set lang "c"
        }
	-help -
	--help {
	    show_usage
	    exit 0
	}
	-* {
	    show_usage
	    puts stderr ""
	    puts stderr "unknown option: $arg"
	    exit 1
	}
        default {
            newclass $arg $hext $cext $lang
        }
    }
}
