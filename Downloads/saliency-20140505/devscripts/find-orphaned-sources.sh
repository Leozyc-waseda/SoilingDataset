#!/bin/sh

# Quick script to try to find orphaned source files (i.e., aren't
# referenced by any other source files).

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/find-orphaned-sources.sh $
# $Id: find-orphaned-sources.sh 6592 2006-05-16 19:46:31Z rjpeters $

scriptdir=`dirname "$0"`
depfile="$scriptdir/../build/alldepends-v2"

if ! test -f $depfile; then
    echo "try 'make build/alldepends-v2' before running $0"
    exit 1
fi

for f in `$scriptdir/list-sources.sh`; do
    f=${f#src/}

    case "$f" in
	*.h|*.hh|*.H|*.hpp)
	    if ! fgrep "$f" "$depfile" > /dev/null; then
		echo $f
	    fi
	    ;;
	*.c|*.cc|*.C|*.cpp|*.Q)
	    sofile="${f%.*}.so"
	    if ! fgrep "$sofile" "$depfile" > /dev/null; then
		echo $f
	    fi
	    ;;
	*)
	    echo "don't know what to do with $f"
	    ;;
    esac
done
