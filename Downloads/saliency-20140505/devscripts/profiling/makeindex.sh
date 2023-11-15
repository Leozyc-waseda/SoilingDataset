#!/bin/sh

# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/makeindex.sh $
# $Id: makeindex.sh 6940 2006-08-04 23:51:08Z rjpeters $

if test $# -ne 1; then
    echo "usage: $0 <data-dir>"
    exit 1
fi

datdir=$1

index=$datdir/index.html

title="profiling data"
if test -f $datdir/TITLE; then
    title=`cat $datdir/TITLE`
fi

rm -f $index

echo "<html>" >> $index
echo "<head><title>$title</title></head>" >> $index
echo "<body>" >> $index

if test -f $datdir/DESCRIPTION; then
    echo "<h2>Summary</h2>" >> $index
    cat $datdir/DESCRIPTION >> $index
    echo "<hr>" >> $index
fi

echo "<h2>Time/space complexity charts and big-O values</h2>" >> $index
if test -f $datdir/stats.png; then
    pngtopnm $datdir/stats.png | pnmscale -xsize 300 | pnmtopng \
	> $datdir/stats.thumb.png
    echo "<a href='stats.png'><img src='stats.thumb.png'><br>overall statistics</a>" >> $index
else
    echo "<a href='stats.png'><b>overall statistics</b></a>" >> $index
fi
echo "<br>" >> $index
echo "<br>" >> $index
echo "<br>" >> $index
echo "<hr>" >> $index

echo "<h2>Call graphs and profiling for individual test runts</h2>" >> $index
echo "<table border='1'>" >> $index
echo "<tr><th>image size</th><th>single-thread</th><th>multi-thread</th></tr>" >> $index

for sz in 5120x3840 3584x2688 2560x1920 1792x1344 1280x960 896x672 640x480 448x336 320x240 224x168 160x120 112x84; do

    echo "<tr><td>$sz</td>" >> $index

    for ext in t0 t15; do

	echo "<td><a href='gprof${sz}-${ext}.html'><img src='gprof${sz}-${ext}.thumb.png'><br>call graph</a></td>" >> $index

    done

    echo "</tr>" >> $index

done

echo "</table>" >> $index
echo "</hr>" >> $index

echo "</body>" >> $index
