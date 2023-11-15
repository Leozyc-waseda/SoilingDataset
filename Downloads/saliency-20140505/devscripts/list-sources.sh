#!/bin/sh

# $Id: list-sources.sh 5791 2005-10-27 20:46:09Z rjpeters $

# Just print a list of all source files to stdout, for use in other
# scripts.

flz=$(find `dirname $0`/../src \
    -name \*.H \
    -or -name \*.C \
    -or -name \*.Q \
    -or -name \*.cc \
    -or -name \*.hh \
    -or -name \*.c \
    -or -name \*.h \
    -or -name \*.hpp \
    -or -name \*.cpp)

for f in $flz; do
    g=${f#`dirname $0`/../}
    echo $g
done
