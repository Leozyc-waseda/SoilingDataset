#!/bin/sh

# $Id: normalize-whitespace.sh 8266 2007-04-18 18:23:38Z rjpeters $

if test $# -lt 1; then
    echo "usage: $0"
    echo "           <srcdir1 or srcfile1> ?<srcdir2 or srcfile2> ...?"
    echo ""
    echo "    Use this program to normalize the whitespace in one or more"
    echo "    files into a uniform format. Two things happen: (1) tabs are"
    echo "    converted to spaces (8 spaces per tab), and (2) any trailing"
    echo "    whitespace on a line is deleted."
    echo ""
    echo "    The program operates on either files or directories (or"
    echo "    both). If you pass it a directory name, it will look for all"
    echo "    C/C++ source files under that directory (including in"
    echo "    subdirectories) with extensions .C, .H, .c, .h, .cc, .hh,"
    echo "    .cpp, or .hpp. If you pass it a file name, it just operates"
    echo "    on that file."
    echo ""
    echo "    As the program runs, it prints the names of those files (and"
    echo "    only those files) that have had their whitespace changed. Thus,"
    echo "    if you run the program a second time on the same directories"
    echo "    or files, you should see no output the second time."
    echo ""
    echo "    If there were no changes as a result of normalizing whitespace"
    echo "    in a given file, then the original file is left untouched (and"
    echo "    its modification timestamp is unchanged)."
    exit 1
fi

for arg in "$@"; do
    if test -d "$arg"; then
	files=`find "$arg" -name \*.[CH] -or -name \*.[ch] -or -name \*.cc -or -name \*.hh -or -name \*.[ch]pp`
    elif test -f "$arg"; then
	files="$arg"
    else
	echo "ERROR: '$arg' appears to be neither a file nor a directory"
	echo "exiting"
	exit 1
    fi

    for f in $files; do

	# Use sed to replace tabs with spaces, and kill trailing
	# whitespace

	perl -p       -e 's/\t/        /g' $f \
	    | perl -p -e 's/  *$//g'       > ${f}.new


	# Now use diff to check if anything changed as a result of sed

	if diff ${f} ${f}.new > /dev/null; then
	    # no change, so be quiet
	    rm ${f}.new
	else

	    # Something changed, so now run diff again, this time
	    # ignoring whitespace, to make sure we haven't messed
	    # anything up

	    if diff -b ${f} ${f}.new > /dev/null; then
	    # looks good
		:
	    else
		diff -bu ${f} ${f}.new
		echo "[arg=$arg] ERROR: normalize-whitespace changed something other than whitespace (see preceding diff)"
		exit 1
	    fi

	    echo "[arg=$arg] whitespace changed in $f"

	    # overwrite the existing file; this is relatively safe
	    # since subversion keeps a backup copy around

	    mv ${f}.new $f
	fi
    done
done
