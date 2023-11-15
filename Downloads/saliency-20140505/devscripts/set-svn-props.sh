#!/bin/sh

# $Id: set-svn-props.sh 4602 2005-06-19 23:40:54Z rjpeters $

# Quick script to set svn:keywords and svn:eol-style properties
# uniformly on all source files.

# Usage: "./devscripts/set-svn-props.sh"
# (no command-line arguments needed).

scriptdir=`dirname $0`;

keywords_val="Author Date Id Revision HeadURL"
eolstyle_val="native"

for f in `$scriptdir/list-sources.sh`; do
    keywords=`svn propget svn:keywords "$f"`
    status=$?
    if test $status -ne 0; then

	# here, 'svn propget' failed (probably because the file is
	# non-versioned), so let's just ignore the file

	echo "ignoring $f"
	echo ""

    elif test "x$keywords" != "x$keywords_val"; then
	svn propset svn:keywords "$keywords_val" "$f"
    fi

    eolstyle=`svn propget svn:eol-style "$f"`
    status=$?
    if test $status -ne 0; then

	# here, 'svn propget' failed (probably because the file is
	# non-versioned), so let's just ignore the file

	echo "ignoring $f"
	echo ""

    elif test "x$eolstyle" != "x$eolstyle_val"; then
	svn propset svn:eol-style "$eolstyle_val" "$f"
    fi

done
