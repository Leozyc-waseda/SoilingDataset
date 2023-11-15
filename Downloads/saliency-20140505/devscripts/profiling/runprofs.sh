#!/bin/sh

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/runprofs.sh $
# $Id: runprofs.sh 6938 2006-08-04 23:05:48Z rjpeters $

if test $# -ne 1; then
    echo "usage: $0 <data-dir>"
    exit 1
fi

dir=`dirname $0`
datdir=$1

for ext in t0 t15; do

    for sz in 5120x3840 3584x2688 2560x1920 1792x1344 1280x960 896x672 640x480 448x336 320x240 224x168 160x120 112x84; do

	$dir/profgraph \
	    $dir/../../bin/ezvision $datdir/gmon${sz}-${ext}.out \
	    http:///ilab.usc.edu/rjpeters/sdoc/html $datdir/gprof${sz}-${ext} &
    done

    wait

done
