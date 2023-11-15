#!/usr/bin/tclsh

## #################################################################### ##
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the ##
## University of Southern California (USC) and the iLab at USC.         ##
## See http:##iLab.usc.edu for information about this project.          ##
## #################################################################### ##
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; filed July 23, 2001, following provisional applications     ##
## No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).##
## #################################################################### ##
## This file is part of the iLab Neuromorphic Vision C++ Toolkit.       ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is free software; you can   ##
## redistribute it and/or modify it under the terms of the GNU General  ##
## Public License as published by the Free Software Foundation; either  ##
## version 2 of the License, or (at your option) any later version.     ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  ##
## that it will be useful, but WITHOUT ANY WARRANTY; without even the   ##
## implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ##
## PURPOSE.  See the GNU General Public License for more details.       ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   ##
## to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   ##
## Boston, MA 02111-1307 USA.                                           ##
## #################################################################### ##
##
## Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
## $Id: extract_templates.tcl 8133 2007-03-19 20:50:56Z rjpeters $
##

##########################################################################

# This script helps with the process of making explicit instantiations
# for a set of template functions. Basically, it does a
# quick-and-dirty parse of a header file that contains template
# function declarations, and then it turns these declarations into a
# series of explicit template function instantiations, which it writes
# to stdout. The functions can be specified for an arbitrary set of
# types, which can be specified on the command-line.

# The basic usage is like this:
#
#   extract_templates Header.H type1 type2 > Header.I

# The idea is that the output can be captured in some intermediate
# file, which can then be #include'd into the .C file where the
# template functions are actually defined.

# Limitations:

# (1) The script expects the "template <class T>" part of the
# declaration to be on its own line, with the rest of the prototype
# following on (multiple) subsequent lines.

# (2) The script will probably choke if there are template class
# definitions in the same header file along with template function
# declarations, although it is "smart" enough to ignore a template
# class forward-declaration (like this: "template <class T> class
# Image")

# (3) The script will only work for template functions that have one
# template parameter, although this is not too much of a limitation
# because it's pretty difficult in general to get the right set of
# explicit instantiations for a template function with multiple
# template parameters.

# (4) The script does NOT do any C-preprocessing, so it may not work
# if the template function declarations contain macros (but macros are
# already evil anyway, so...)

##########################################################################


# This is the magic word, which, if seen as a template-id in a
# template function declaration, will cause that function to be
# instantiated with "color" types in addition to the regular types.

set COLOR_MAGIC "T_or_RGB"

proc compute_typelist { template_id typelist color_typelist } {

    # Procedure to compute a typelist given a template_id (i.e., the
    # template_id is the "T" in a declaration like "template <class
    # T>...")

    # First of all, the default is to use the one that was passed to
    # us...
    set result_typelist $typelist

    # ... but if the template-id matches ::COLOR_MAGIC, then we use
    # color types as well
    if { [string equal $template_id $::COLOR_MAGIC] } {
	set result_typelist $color_typelist
    } elseif { [string first "T_" $template_id] == 0 } {

	# ... otherwise if the template-id matches "T_*", then we
	# parse the characters following the "T_". Each character is
	# treated as a request for a particular instantiation (see the
	# switch statement below). For example, "T_fB" requests
	# instantiations for 'float' and for 'PixRGB<byte>'. This
	# mechanism is extensible; additional character codes can be
	# added to the switch statement below as needed.

	set result_typelist [list]
	for {set i 2} {$i < [string length $template_id]} {incr i} {
	    set c [string index $template_id $i]
	    switch $c {
		f {lappend result_typelist " float "}
		F {lappend result_typelist " PixRGB<float> "}
		b {lappend result_typelist " byte "}
		B {lappend result_typelist " PixRGB<byte> "}
		i {lappend result_typelist " int "}
		I {lappend result_typelist " PixRGB<int> "}
		default {error "unknown typecode '$c'"}
	    }
	}
    }

    return $result_typelist
}

proc split_param_list { param_list } {

    # Split a function's parameter list (taken as a string) into a Tcl
    # list where each element is one of the parameter
    # declarations. This is essentially splitting the paramlist on
    # commas (","), but we have to taken into account that nested
    # commas might occur inside ()'s or <>'s.

    set chars [split $param_list {}]

    set params [list]

    set current_param ""

    set nest_depth 0
    set inquote 0

    foreach c $chars {
	switch -exact -- $c {
	    "<" -
	    "(" { if { !$inquote } {incr nest_depth} }

	    ")" -
	    ">" { if { !$inquote } {incr nest_depth -1} }

	    "\'" -
	    "\"" { set inquote [expr !$inquote] }

	    "," {
		if { !$inquote && $nest_depth == 0 } {
		    lappend params $current_param
		    set current_param ""
		    continue
		}
	    }
	}
	append current_param $c
    }

    if { $inquote } {
	error "param list parsing ending in a quote '${param_list}'"
    }

    lappend params $current_param

    return $params
}

proc subst_template_param { decl template_id actual_type } {

    # Replace all instances of the template formal parameter (i.e.,
    # "T") with the actual parameter (i.e., "byte" or "float").
    regsub -all \
	[format {\m%s\M} $template_id] \
	$decl \
	$actual_type \
	new

    return $new
}

proc subst_INST_CLASS { decl } {

    # Prepend the macro INST_CLASS before the function name (that is,
    # the token that precedes the first open parentheses)
    regsub \
	{\m(\w*)\M\(} \
	$decl \
	[format {INST_CLASS \1%s} "("] \
	new

    return $new
}

proc subst_static_keyword { decl } {

    # Eliminate any uses of the "static" keyword, since this does not
    # carry over the same meaning from declaration to decl
    regsub -all \
	{\s*\mstatic\M\s*} \
	$decl \
	" " \
	new

    return $new
}

proc subst_default_params { decl } {

    # Remove any default parameter values from each parameter
    # declaration, since these are not allowed as part of an explicit
    # instantiation (i.e. change "const int x = 0" to "const int x").

    # First split out the entire parameter list en masse.
    if { [regexp \
	      {([^\(]*)\((.*)\)([^\)]*)} \
	      $decl \
	      fullmatch \
	      head \
	      paramlist \
	      tail] != 1 } {
	error "couldn't parse decl '${decl}'"
    }

    # Now split the full parameter list into individual params.
    set params [split_param_list $paramlist]

    set new_params [list]

    # Now remove any default value from each param.
    foreach param $params {

	regsub \
	    {=.*} \
	    $param \
	    "" \
	    new

	lappend new_params $new
    }

    # Finally, reconstruct the function declaration.
    set new_paramlist [join $new_params ","]

    return "${head}($new_paramlist)${tail}"
}

proc subst_typename_keyword { decl } {

    # Eliminate any uses of the "typename" keyword, since this does
    # not carry over the same meaning from declaration to
    # instantiation
    regsub -all \
	{\s*\mtypename\M\s*} \
	$decl \
	" " \
	new

    return $new
}

proc fixup_decl { decl } {

    # Apply all necessary transformations to turn a header-file-style
    # declaration into an explicit-instantiation-style declaration.

    set decl [subst_INST_CLASS $decl]
    set decl [subst_static_keyword $decl]
    set decl [subst_default_params $decl]
    set decl [subst_typename_keyword $decl]

    return $decl
}

proc get_declarations { fname typelist } {
    set chan [open $fname r]

    set buf ""

    set decls [list]

    set template_id ""

    # Build up a typelist that also includes "color" types; i.e. types
    # of the form PixRGB<T>. We use this color typelist for
    # instantiation in cases where the template-id in the declaration
    # is the magic word matching ::COLOR_MAGIC; in all other cases we
    # assume that color types should not be explicitly instantiated.
    set color_typelist [list]

    foreach type $typelist {
	lappend color_typelist $type

	# The extra spaces on either side are to avoid problems with
	# spurious '>>' tokens (e.g. Image<PixRGB<byte>> is wrong, but
	# Image< PixRGB<byte> > is ok).
	lappend color_typelist " PixRGB<$type> "

	# We don't unequivocally instantiate functions with the H2SV1
	# and H2SV2 pixel types, because only a tiny fraction of
	# functions are actually used with these types, so we can keep
	# compile times down by just explicitly requesting only those
	# instantiations that are needed

	#lappend color_typelist " PixH2SV1<$type> "
	#lappend color_typelist " PixH2SV2<$type> "

	# For now, don't instantiate functions for the following pixel
	# types -- nobody is currently using Image with this pixel
	# types, so we can save a good deal of compile time by not
	# instantiating functions with these types. NOTE: the list of
	# which types to instantiate should be kept in sync with the
	# macros in PixelsInst.H -- types listed in PIX_INST should be
	# added to color_typelist above, while types listed only in
	# PIX_INST_ALL can remain commented-out in the list below:

	#lappend color_typelist " PixHSV<$type> "
	#lappend color_typelist " PixYUV<$type> "
	#lappend color_typelist " PixYIQ<$type> "
	#lappend color_typelist " PixH2SV3<$type> "
	#lappend color_typelist " PixHyper<$type,3> "
    	#lappend color_typelist " PixHyper<$type,4> "
    }

    while { [gets $chan line] != -1 } {

	# See if the line is the start of a template declaration, with
	# something of the form "template <class T>"; we extract the
	# formal template parameter ("T" in this example) so that we
	# can substitute the actual template parameters (like "byte"
	# or "float") later on when we form the explicit
	# instantiations.
	if { [regexp {^\s*template\s*<(class|typename)\s*\m(\w*)\M\s*>} \
		  $line fullmatch sub1 sub2] } {
	    if { [string length $buf] != 0 } {
		puts stderr "buf: $buf"
		puts stderr "line: $line"
		error "Found a new 'template' declaration when \
the previous declaration was not completed"
	    }

	    # If the putative declaration contains 'inline' or
	    # contains {} brackets, then we can skip it since we don't
	    # need explicit instantiations of inline functions:
	    if { [regexp {\minline\M} $line] || [regexp {[\{\}]} $line] } {
		set buf ""
		continue
	    }

	    set template_id $sub2

	    set buf "template"

	} elseif { [string length $buf] > 0 } {
	    append buf " " [string trim $line]
	}

	# If the putative declaration contains 'inline' or contains {}
	# brackets, then we can skip it since we don't need explicit
	# instantiations of inline functions:
	if { [regexp {\minline\M} $buf] || [regexp {[\{\}]} $buf] } {
	    set buf ""
	    continue
	}

	# If the line ends with a semicolon, then we have found the
	# presumptive end of the template function declaration, so add
	# it to the list of declarations, and clear our buffer to get
	# ready for the next declaration
	if { [regexp {;\s*$} $line] } {

	    # Check to see if the declaration contains a pair of
	    # parentheses; if it does not, then it cannot have been a
	    # function declaration (in this case it was probably a
	    # forward class declaration like "template <class T> class
	    # Image"
	    if { [regexp {\(.*\)} $buf] } {

		set current_typelist \
		    [compute_typelist \
			 $template_id $typelist $color_typelist]

		set decl [fixup_decl $buf]

		foreach type $current_typelist {
		    lappend decls [subst_template_param \
				       $decl $template_id $type]
		}
	    }
	    set buf ""
	}
    }

    close $chan

    return $decls
}

proc wrap_lines { pfx max_length  text} {
    set words [split $text]

    set current_line ""

    set lines [list]

    foreach word $words {
	if { [string length $current_line] == 0 } {

	    # Very first word
	    set current_line "$pfx$word"
	} elseif { 1 + [string length $word] \
		 + [string length $current_line] \
		 <= $max_length } {

	    # add another word to the current line
	    append current_line " $word"
	} else {

	    # finish the current line ...
	    lappend lines $current_line

	    # ... and start a new line
	    set current_line "$pfx$word"
	}
    }

    lappend lines $current_line

    return $lines
}

#
# The main script:
#

set fname [lindex $argv 0]

set types [lrange $argv 1 end]

if {[llength $types] == 0} { set types "byte int float" }

puts "/*               -*- mode: c++ -*-"
puts ""
puts "   DO NOT EDIT THIS FILE -- CHANGES WILL BE LOST!"
puts ""

puts [join [wrap_lines "   " 76 \
		"\[[clock format [clock seconds]]\] This file was automatically generated by applying the script \"$::argv0\" to the template declarations in source file \"$fname\" for types \[[join $types {, }]\]"] \
	  "\n"]

puts "*/"

# Oddly, template functions in a class require a different syntax than
# template functions in a namespace. For those in a class, we need to
# prefix "ClassName::" before the *name* of each function in its
# explicit instantiation. For those in a namespace, that syntax
# doesn't work; instead we have to wrap all the explicit
# instantiations in a "namespace NamespaceName {}" block.

puts "\#ifndef INST_CLASS"
puts "\#  define INST_CLASS"
puts "\#endif"
puts "\#ifdef INST_NAMESPACE"
puts "namespace INST_NAMESPACE {"
puts "\#endif"
puts [join [get_declarations $fname $types] "\n"]
puts "\#ifdef INST_NAMESPACE"
puts "}"
puts "\#endif"
