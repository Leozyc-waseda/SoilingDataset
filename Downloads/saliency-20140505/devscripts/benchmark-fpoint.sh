#!/bin/sh

# $Id: benchmark-fpoint.sh 4327 2005-06-07 18:37:16Z rjpeters $

dir=`dirname $0`
log=$dir/benchmark-fpoint.log

if test -e $log; then
    mv $log ${log}.old
fi

lines=2
for opt in -O0 -O1 -O2 -O3; do
    src=$dir/benchmark-fpoint.cc
    exe=$dir/benchmark-fpoint-$opt
    g++ $opt $src -o $exe
    $exe 30 1024 $opt | tail -$lines | tee -a $log
    rm $exe
    lines=1
done
