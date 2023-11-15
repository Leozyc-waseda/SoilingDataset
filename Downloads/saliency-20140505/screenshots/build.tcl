#!/usr/bin/tclsh

# $Id: build.tcl 7865 2007-02-08 02:12:38Z itti $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/screenshots/build.tcl $

source [file dirname [info script]]/../tests/config.tcl

set TMPCOUNTER 0

proc tmpfilename {} {
    return "demo_tmp_[pid]_[incr ::TMPCOUNTER]"
}

proc contents_of { fname } {
    if { ![file exists $fname] } { return "" }
    set fd [open $fname r]
    set text [read $fd]
    close $fd
    return $text
}

proc thumbnail_name { num } {
    return [format "%s/thumbnails/demo%03d.png" \
		[file dirname [info script]] $num]
}

proc screenshot_name { num } {

    set ext [file extension $::DEMO($num,screenshot)]

    switch -exact -- $ext {
	.pnm {
	    return [format "%s/screenshots/demo%03d.png" \
			[file dirname [info script]] $num]
	}
	.pfm {
	    return [format "%s/screenshots/demo%03d.png" \
			[file dirname [info script]] $num]
	}
	.mpg {
	    return [format "%s/screenshots/demo%03d.mpg" \
			[file dirname [info script]] $num]
	}
    }

    puts "oops: bogus screenshot extension $ext"
    exit 1
}

proc wrap { text length } {
    set result [list]
    set line ""
    foreach word [split $text " "] {
	append line $word " "
	if { [string length $line] > $length } {
	    lappend result $line
	    set line ""
	}
    }
    if { [string length $line] > 0 } {
	lappend result $line
    }
    return [join $result "\n"]
}

proc load_defs_file { fname } {

    set ::DEFSFILE $fname

    set fd [open $fname r]
    set buf ""
    set field ""
    set demonumber 0
    while { [gets $fd line] >= 0 } {

	if { [string equal $line "@newdemo"] } {

	    if { [string length $field] > 0 && \
		     [string length $buf] > 0 } {
		set ::DEMO($demonumber,$field) [string trim $buf]

		set buf ""
		set field ""
	    }

	    incr demonumber

	} elseif { [regexp {^@(.*)} $line - newfield] } {

	    if { [string length $field] > 0 && \
		     [string length $buf] > 0 } {
		set ::DEMO($demonumber,$field) [string trim $buf]
	    }

	    set buf ""
	    set field $newfield

	} elseif { [regexp {^[ \t]+(.*)} $line - words] } {

	    append buf $words " "
	}

    }

    if { [string length $field] > 0 && \
	     [string length $buf] > 0 } {
	set ::DEMO($demonumber,$field) [string trim $buf]
    }

    close $fd

    set ::NDEMO $demonumber
}

proc generate_html_1 { fd n } {
    set goal $::DEMO($n,goal)
    set command $::DEMO($n,command)
    set outputs $::DEMO($n,outputs)
    set notes $::DEMO($n,notes)

    puts $fd ""
    puts $fd ""
    puts $fd [format "<!-- demo #%d -->" $n]
    puts $fd [format "<!-- %s -->" [clock format [clock seconds]]]
    puts $fd [format "<!-- %s -->" {$Id: build.tcl 7865 2007-02-08 02:12:38Z itti $}]
    puts $fd ""
    puts $fd {<tr>}
    puts $fd {    <td align="center" style="background-color: #bbddff;" valign="middle">}
    puts $fd [format {        <a href="%s"><img src="%s" style="border-style: none;" alt="%s"></a>} \
		  [screenshot_name $n] [thumbnail_name $n] "screenshot for demo #$n"]
    puts $fd {    </td>}
    puts $fd {    <td style="background-color: #bbddff;">}
    puts $fd {        <table border="0">}
    puts $fd {            <tbody>}
    puts $fd {                <tr>}
    puts $fd {                    <td valign="middle"><b>Goal:</b></td>}
    puts $fd {                    <td valign="middle">}
    puts $fd [format "<!-- demo #%d goal -->" $n]
    puts $fd [wrap $goal 50]
    puts $fd {                    </td>}
    puts $fd {                </tr>}
    puts $fd {                <tr>}
    puts $fd {                    <td valign="middle"><b>Command:</b></td>}
    puts $fd {                    <td valign="middle" style="font-family: monospace;">}
    puts $fd [format "<!-- demo #%d command -->" $n]
    puts $fd $command
    puts $fd {                    </td>}
    puts $fd {                </tr>}
    puts $fd {                <tr>}
    puts $fd {                    <td valign="middle"><b>Outputs:</b></td>}
    puts $fd {                    <td valign="middle" style="font-family: monospace;">}
    puts $fd [format "<!-- demo #%d outputs -->" $n]
    puts $fd [wrap $outputs 50]
    puts $fd {                    </td>}
    puts $fd {                </tr>}
    puts $fd {                <tr>}
    puts $fd {                    <td valign="middle"><b>Notes:</b></td>}
    puts $fd {                    <td valign="middle">}
    puts $fd [format "<!-- demo #%d notes -->" $n]
    puts $fd [wrap $notes 50]
    puts $fd {                    </td>}
    puts $fd {                </tr>}
    puts $fd {            </tbody>}
    puts $fd {        </table>}
    puts $fd {    </td>}
    puts $fd {</tr>}
    puts $fd ""
    puts $fd ""
}

proc generate_html { fd } {
    puts $fd {<table border="0" cellpadding="2" cellspacing="2" width="100%">}
    for {set i 1} {$i <= $::NDEMO} {incr i} {
	generate_html_1 $fd $i
    }
    puts $fd {</table>}
}

proc generate_runner { fd } {
    puts $fd "#!/bin/sh"
    puts $fd ""

    puts $fd "echo 'available demos:'"
    puts $fd "echo ''"

    for {set i 1} {$i <= $::NDEMO} {incr i} {
	puts $fd [format "echo '%3d)\t$::DEMO($i,goal)'" $i]
    }

    puts $fd "echo ''"
    puts $fd "echo 'which demo would you like to run (1-$::NDEMO)? '"
    puts $fd "read number"

    puts $fd "case \$number in"
    for {set i 1} {$i <= $::NDEMO} {incr i} {
	puts $fd "\t${i})"
	set cmd [string map "ezvision [config::exec_prefix]/bin/ezvision" \
		     $::DEMO($i,command)]
	puts $fd "\t\techo 'ok, demo number ${i} is:'"
	puts $fd "\t\techo '\t$cmd'"
	puts $fd "\t\techo ''"
	puts $fd "\t\techo 'press return to run demo number $i'"
	puts $fd "\t\techo '(once the demo is running, you can press ctrl-c to stop it)'"
	puts $fd "\t\tread dummy"
	puts $fd "\t\texec $cmd"
	puts $fd "\t\t;;"
    }
    puts $fd "\t*)"
    puts $fd "\t\techo 'invalid demo number' \$number"
    puts $fd "\t\t;;"
    puts $fd "esac"
}

proc generate_screenshots {} {
    for {set i 1} {$i <= $::NDEMO} {incr i} {
	set orig $::DEMO($i,command)
	set command $orig
	set command [regsub -all -- {--out=([^ ]+) ?} $command ""]
	set command [string map "ezvision [config::exec_prefix]/bin/ezvision" \
			 $command]
	set args $::DEMO($i,screenshotargs)

	set demopfx "tmp-[pid]-demo-${i}-"

	set args [string map "DEMOPREFIX $demopfx" $args]

	set thumbnail $::DEMO($i,thumbnail)
	set screenshot $::DEMO($i,screenshot)

	set thumbnail [string map "DEMOPREFIX $demopfx" $thumbnail]
	set screenshot [string map "DEMOPREFIX $demopfx" $screenshot]

	puts "command:  $command $args"

	set code [catch {eval exec $command $args 2>@ stdout} result]

	if { $code != 0 } {
	    puts "oops: '$command $args' failed:"
	    puts $result
	    exit 1
	}

	if { ![file exists $thumbnail] } {

	    puts "oops: $thumbnail not generated!"
	    exit 1
	}

	if { ![file exists $screenshot] } {

	    puts "oops: $screenshot not generated!"
	    exit 1
	}

	file mkdir [file dirname [thumbnail_name $i]]
	set ext [file extension $thumbnail]
	set thumb $thumbnail

	if { [string equal $ext ".pfm"] } {
	    set thumb "[tmpfilename].pgm"
	    set errfile [tmpfilename]
	    set convcmd "[config::exec_prefix]/bin/pfmtopgm"

	    set code [catch {
		exec $convcmd $thumbnail $thumb 2> $errfile} result]

	    set errmsg [contents_of $errfile]
	    file delete -force $errfile

	    if { $code != 0 } {
		puts $errmsg
		puts $result
		exit 1
	    }
	}

	set errfile [tmpfilename]

	set code [catch {
	    exec pnmscale -xsize 200 $thumb \
		| pnmtopng \
		> [thumbnail_name $i] 2> $errfile} result]

	set errmsg [contents_of $errfile]
	file delete -force $errfile

	if { $code != 0 } {
	    puts $errmsg
	    puts $result
	    exit 1
	}

	puts "thumbnail: $thumbnail -> [thumbnail_name $i]"

	file mkdir [file dirname [screenshot_name $i]]
	set ext [file extension $screenshot]
	switch -exact -- $ext {
	    .pnm {

		set errfile [tmpfilename]

		set code [catch {
		    exec pnmtopng $screenshot \
			> [screenshot_name $i] 2> $errfile} result]

		set errmsg [contents_of $errfile]
		file delete -force $errfile

		if { $code != 0 } {
		    puts $errmsg
		    puts $result
		    exit 1
		}
	    }
	    .pfm {
		set errfile [tmpfilename]
		set convcmd "[config::exec_prefix]/bin/pfmtopgm"

		# NOTE: pfmtopgm can save .png images (based on extension):
		set code [catch {
		    exec $convcmd $screenshot \
			[screenshot_name $i] 2> $errfile} result]

		set errmsg [contents_of $errfile]
		file delete -force $errfile

		if { $code != 0 } {
		    puts $errmsg
		    puts $result
		    exit 1
		}
	    }
	    .mpg {
		file rename -force $screenshot [screenshot_name $i]
	    }
	    default {
		puts "oops: bogus screenshot extension $ext"
		exit 1
	    }
	}

	puts "screenshot: $screenshot -> [screenshot_name $i]"

	foreach f [glob -nocomplain "${demopfx}*.pnm"] {
	    file delete $f
	}

	puts ""
    }
}

load_defs_file [file dirname [info script]]/defs.txt

foreach action $argv {
    switch -exact -- $action {
	html {
	    set file "[file dirname [info script]]/index.html"
	    set fd [open $file w]

	    puts $fd {<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"}
	    puts $fd {  "http://www.w3.org/TR/html4/strict.dtd">}
	    puts $fd {<html>}
	    puts $fd {<head>}
	    puts $fd {<title>Screenshots - iLab Neuromorphic Vision C++ Toolkit (iNVT)</title>}
	    puts $fd {</head>}
	    puts $fd {<body>}

	    generate_html $fd

	    puts $fd {</body>}
	    puts $fd {</html>}

	    close $fd
	    puts "generated html file $file"
	}
	htmlraw {
	    set file "[file dirname [info script]]/index.raw"
	    set fd [open $file w]

	    generate_html $fd

	    close $fd
	    puts "generated html fragment $file"
	}
	screenshots {
	    generate_screenshots
	}
	runner {
	    set script "[file dirname [info script]]/rundemo.sh"
	    set fd [open $script w]
	    generate_runner $fd
	    close $fd
	    file attributes $script -permissions u+x
	    puts "generated demo-runner script $script"
	}
    }
}
