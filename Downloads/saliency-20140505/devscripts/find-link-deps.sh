#!/bin/sh

# $Id: find-link-deps.sh 8193 2007-03-29 22:38:04Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/find-link-deps.sh $

# find all the source files that must be linked to a particular source
# file; this requires that 'make ldeps' has already been run to
# generate an ldeps file

if test $# -ne 2; then
    echo "usage: $0 ldepsfile srcname"
    echo ""
    echo "This program finds all the source files that are link dependencies"
    echo "of a given source file; you must first run 'make ldeps' and pass"
    echo "the name of the generated ldeps file as the first argument here."
    echo ""
    echo "For example (after having run make ldeps):"
    echo ""
    echo "    wc \`$0 ldeps src/INVT/ezvision.C\`"
    echo ""
    echo "will compute all of the link dependencies of ezvision and run"
    echo "all of those source files through wc."
    exit 1
fi

ldepsfile=$1
srcname=$2

for f in $(fgrep $srcname $ldepsfile | awk '{print $2}'); do
    case $f in
	*.cc|*.Q|*.C|*.c|*.cpp)
	    stem=${f%.*}
	    if test -f ${stem}.h; then
		echo ${stem}.h
	    elif test -f ${stem}.H; then
		echo ${stem}.H
	    elif test -f ${stem}.hpp; then
		echo ${stem}.hpp;
	    fi
	    ;;
    esac
    echo $f
done
