#!/bin/sh

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/runtests.sh $
# $Id: runtests.sh 6937 2006-08-04 22:34:59Z rjpeters $

#opts="--mem-stats --out=none --sv-type=None"
opts="--mem-stats --out=none --sv-type=None --maxnorm-type=Maxnorm"

dir=`dirname $0`

for sz in 5120x3840 3584x2688 2560x1920 1792x1344 1280x960 896x672 640x480 448x336 320x240 224x168 160x120 112x84; do

    if ! test -f movie/${sz}/000000.pnm; then
	mkdir -p movie/${sz}
	$dir/../../bin/stream --in=tests/inputs/mpegclip1.mpg --rescale-input=$sz --out=pnm movie/${sz}/
    fi

    if ! test -f log${sz}-t0.txt; then

	(time $dir/../../bin/ezvision --in=movie/${sz}/#.pnm $opts) \
	    > log${sz}-t0.txt 2>&1

	chmod -w log${sz}-t0.txt
	chmod -w gmon.out
	mv gmon.out gmon${sz}-t0.out || exit 1

    fi

    if ! test -f log${sz}-t15.txt; then

	(time $dir/../../bin/ezvision --in=movie/${sz}/#.pnm --vc-type=J:15CIOFM $opts) \
	    > log${sz}-t15.txt 2>&1

	chmod -w log${sz}-t15.txt
	chmod -w gmon.out
	mv gmon.out gmon${sz}-t15.out || exit 1

    fi

done
