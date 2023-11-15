#!/bin/sh

# $Id: strip-doxygen-markup.sh 6254 2006-02-17 17:53:49Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/strip-doxygen-markup.sh $

# This is a simple script to strip doxygen markup out of a file,
# making it more readable as plain text. The stripped text is written
# to standard output.

if test $# -ne 1; then
    echo "usage: $0 <input-file>"
    exit 1
fi

sed \
    -e 's/\\section/[section]/g' \
    -e 's/\\subsection/[subsection]/g' \
    -e 's/\\subsubsection/[subsubsection]/g' \
    -e 's/\\code//g' \
    -e 's/\\endcode//g' \
    -e 's/\\verbatim//g' \
    -e 's/\\endverbatim//g' \
    -e 's/\\note/NOTE:/g' \
    -e 's/\\par//g' \
    -e 's/\\ref//g' \
    -e 's/<!--//g' \
    -e 's/-->//g' \
    -e 's/<ul>//g' \
    -e 's/<\/ul>//g' \
    -e 's/<li> /* /g' \
    -e 's/<li>/* /g' \
    -e 's/<\/li>//g' \
    $1
