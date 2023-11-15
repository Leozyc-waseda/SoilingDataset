#!/bin/sh
#
# USAGE: mpg2divx.sh <movie1.mpg> ... <movieN.mpg>
# will convert the movies to DivX .avi files

for f in $*; do
		echo "test $f";
		ff=`echo $f | sed -e "s/mpg\$/avi/"`
    if [ ! -f $ff ]; then
        echo "##### Creating $ff"
        mencoder -noskip -ovc lavc -lavcopts vcodec=mpeg4:vhq -o $ff $f
    else
        echo "##### Already have $ff"
    fi
done

# for higher quality, try:
#        mencoder -noskip -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=1800:vhq \
#            -o $ff $f
