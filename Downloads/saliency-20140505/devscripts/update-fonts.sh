#!/bin/sh

# $Id: update-fonts.sh 8078 2007-03-08 20:01:35Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/update-fonts.sh $

# This script is used to (re)generate font header files in src/Image/
# from the corresponding font image files in etc/fonts.

# To run the script, just do ./devscripts/update-fonts.sh from the
# main toolkit directory.

# To add a new font: (1) create a new font image and put it in
# etc/fonts, (2) add your font name to the list of fonts below, and
# (3) re-run this script.

fonts="6x10 7x13 8x13bold 9x15bold 10x20 11x22 12x22 14x26 15x28 16x29 20x38"

if ! test -x bin/font2c; then
    echo "You must run 'make bin/font2c first'"
    exit 1
fi

for f in $fonts; do
    if ! test -f etc/fonts/$f.png; then
	echo "Oops: etc/fonts/$f.png is missing"
	exit 1
    fi

    echo -n "reading etc/fonts/font$f.png ... "
    ./bin/font2c etc/fonts/$f.png font$f \
	> src/Image/font$f.h.tmp \
	2> /dev/null \

    if test $? -ne 0; then
	echo "Oops: bin/font2c failed"
	exit 1
    fi

    if ! test -f src/Image/font$f.h; then
	mv src/Image/font$f.h.tmp src/Image/font$f.h
	echo "src/Image/font$f.h was CREATED"
    elif diff -q src/Image/font$f.h.tmp src/Image/font$f.h > /dev/null; then
	rm -f src/Image/font$f.h.tmp
	echo "src/Image/font$f.h was unchanged"
    else
	mv src/Image/font$f.h.tmp src/Image/font$f.h
	echo "src/Image/font$f.h was UPDATED"
    fi
done
