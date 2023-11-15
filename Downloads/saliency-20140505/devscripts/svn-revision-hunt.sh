#!/bin/sh

# This is a quick script to find the first time (within a specific
# revision range, if desired) that a given regular expression appeared
# in a named source file. The program uses a binary search on revision
# numbers. In order for that binary search to give correct results, it
# is assumed (1) that the pattern MATCHES the file at the end of the
# revision range, (2) that the pattern DOES NOT match the file at the
# start of the revision range, and (3) that the pattern changes from
# non-matching to matching JUST ONCE throughout the revision range
# (i.e., once it appears it stays there and doesn't flip on and
# off). If a target revision is found, then the program prints out the
# log message from that revision (see example below).

# Usage: ./devscripts/svn-revision-hunt.sh <regexp> <filename> [rev1:rev2]

# For example, to find when paramFpuPrecision first appeared in
# Component/ModelManager.C:

# ./devscripts/svn-revision-hunt.sh paramFpuPrecision src/Component/ModelManager.C

# gives the following output:

#      pattern found in r5938 [searching r0:5938]
#  pattern NOT found in r2969 [searching r2969:5938]
#  pattern NOT found in r4453 [searching r4453:5938]
#  pattern NOT found in r5195 [searching r5195:5938]
#  pattern NOT found in r5566 [searching r5566:5938]
#  pattern NOT found in r5752 [searching r5752:5938]
#      pattern found in r5845 [searching r5752:5845]
#      pattern found in r5798 [searching r5752:5798]
#      pattern found in r5775 [searching r5752:5775]
#      pattern found in r5763 [searching r5752:5763]
#  pattern NOT found in r5757 [searching r5757:5763]
#  pattern NOT found in r5760 [searching r5760:5763]
#  pattern NOT found in r5761 [searching r5761:5763]
#  pattern NOT found in r5762 [searching r5762:5763]
#
# >>> in file 'src/Component/ModelManager.C'
# >>> within the search range 0:5938,
# >>> 'paramFpuPrecision' first appeared in revision 5763, with log entry:
# ------------------------------------------------------------------------
# r5763 | rjpeters | 2005-09-17 09:45:34 -0700 (Sat, 17 Sep 2005) | 7 lines
# Changed paths:
#    M /trunk/saliency/src/Component/GlobalOpts.C
#    M /trunk/saliency/src/Component/GlobalOpts.H
#    M /trunk/saliency/src/Component/ModelManager.C
#    A /trunk/saliency/src/Util/fpu.C
#    A /trunk/saliency/src/Util/fpu.H
#
# * Added Util/fpu.[HC] with functions to control the precision and
#   rounding-mode bits of the x87 floating-point control word.
#
# * Added --fpu-precision and --fpu-rounding command-line
#   options. Default behavior is the same as always
#   (--fpu-precision=extended and --fpu-rounding=nearest).


# $Id: svn-revision-hunt.sh 8504 2007-06-19 20:37:13Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/svn-revision-hunt.sh $

if test $# -lt 2; then
    echo "usage: $0 <search-regexp> <filename> [rev1:rev2]"
    exit 1
fi

expr="$1"
f=$2

if test $# -ge 3; then
    rev1=$(echo $3 | cut -d : -f 1)
    rev2=$(echo $3 | cut -d : -f 2)
else
    headrev=$(svn info $f | fgrep Revision | egrep -o "[0-9]+")

    rev1=0
    rev2=$headrev
fi

case $f in
    svn://*)
	if ! svn ls -r$rev2 $f@$rev2 > /dev/null; then
	    exit 1
	fi
	;;
    *)
	if ! test -f $f; then
	    echo "no such file: $f"
	    exit 1
	fi
	;;
esac

lastwithout=$rev1
firstwith=$rev2

curr=$firstwith
firstwith_stat=1

while test $curr -gt 1; do
    suff=""
    case $f in
	svn://*)
	    suff="@$curr"
	;;
    esac

    svn cat -r$curr $f$suff 2> /dev/null | egrep -- "$expr" > /dev/null
    stat=$?
    if test $stat -eq 0; then
        firstwith=$curr
        firstwith_stat=$stat
	printf "     pattern found in       %5s [searching r%d:%d]\n" "r$curr" $lastwithout $firstwith
    else
        lastwithout=$curr
	printf " pattern NOT found in %5s       [searching r%d:%d]\n" "r$curr" $lastwithout $firstwith
    fi
    oldcurr=$curr
    let "curr=(firstwith+lastwithout)/2"
    if test $oldcurr -eq $curr; then
        echo ""
        echo ">>> in file '$f'"
        echo ">>> within the search range ${rev1}:${rev2},"
        if test $firstwith_stat -eq 0; then
            echo ">>> '$expr' first appeared in revision $firstwith, with log entry:"
            svn log --verbose -r$firstwith
        else
            echo ">>> '$expr' does not appear at the end of the search range"
        fi
        exit 0
    fi
done
