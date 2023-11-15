#!/bin/bash

# $Id: make-ldep-html.sh 13884 2010-09-04 09:14:21Z itti $

# Quick script to build a set of html pages showing the link
# dependencies, with clickable directed graphs made by 'dot'. You
# probably don't want to call this directly; instead use 'make
# ldep-html'.

if test $# -ne 3; then
    echo "usage: `basename $0` ldeps-file srcdir outdir"
    exit 1
fi

function generate_html () {
    local dotfilename=$1
    local mapname=$2
    local outdir=$3

    ### (1) generate a png file containing the graph

    dot -Tpng $dotfilename > $outdir/$mapname.png


    ### (2) generate a client-side image map file that contains the
    ### bounding boxes of the clickable regions in our graph image

    # We have to check our dot version in order to select between
    # -Tcmapx and -Tcmap; -Tcmapx is newer and supported only in 2.x,
    # while -Tcmap is deprecated now but is our only option for 1.x:

    rm -f $outdir/$mapname.map

    case $dotversion in
        1.*)
	    echo "<!-- used -Tcmap to generate client-side image map file -->" \
		>> $outdir/$mapname.map
            echo "<map id=\"$mapname\" name=\"$mapname\">" >> $outdir/$mapname.map
            dot -Tcmap $dotfilename >> $outdir/$mapname.map
            echo "</map>" >> $outdir/$mapname.map
            ;;
        *)
	    echo "<!-- used -Tcmapx to generate client-side image map file -->" \
		>> $outdir/$mapname.map
            dot -Tcmapx $dotfilename >> $outdir/$mapname.map
            ;;
    esac


    ### (3) generate a trivial html file that contains the graph image
    ### plus the contents of the image map

    rm -f $outdir/$mapname.html

    usemap=`echo $mapname | tr - _`

    echo "<!-- build date: `date` -->"     >> $outdir/$mapname.html
    echo "<!-- build host: `hostname` -->" >> $outdir/$mapname.html
    echo "<!-- dot -V: `dot -V 2>&1` -->"  >> $outdir/$mapname.html
    echo "<html>"                          >> $outdir/$mapname.html
    echo "<head>"                          >> $outdir/$mapname.html
    echo "<title>$outdir/$mapname</title>" >> $outdir/$mapname.html
    echo "</head>" 			   >> $outdir/$mapname.html
    echo "<body>"  			   >> $outdir/$mapname.html
    echo "<img border=\"0\" "              >> $outdir/$mapname.html
    echo "     src=\"$mapname.png\""       >> $outdir/$mapname.html
    echo "     usemap=#$usemap>"           >> $outdir/$mapname.html
    cat $outdir/$mapname.map               >> $outdir/$mapname.html
    echo "</body>"                         >> $outdir/$mapname.html
    echo "</html>"                         >> $outdir/$mapname.html
}

### main program:

ldeps=$1
srcdir=$2
outdir=$3
thisdir=`dirname $0`

dotversion=`dot -V 2>&1 | cut -d " " -f 3`
echo "dot version $dotversion"

mkdir -p $outdir

echo "modules ..."
$thisdir/dot-ldep-modules.tcl $ldeps \
    | tred \
    > tmp-ldep.dot

generate_html tmp-ldep.dot modules $outdir

rm -f $outdir/index.html
ln -s modules.html $outdir/index.html

rm tmp-ldep.dot

echo "... done"

for d in ${srcdir}*; do
    if test -d $d; then

        echo "directory $d ..." ;

        stem=${d#${srcdir}}

	tmpldepfile=tmp-ldeps-$$-$stem

	# prefilter the ldeps file so that dot-ldep-internal can run a
	# lot faster:
	grep "${d}.*src/" $ldeps > $tmpldepfile

        $thisdir/dot-ldep-internal.tcl \
            $d $tmpldepfile \
            | tred \
            > tmp-ldep-$stem.dot

	rm $tmpldepfile

        generate_html tmp-ldep-$stem.dot $stem $outdir

        rm tmp-ldep-$stem.dot;

        echo "... done";
    fi
done
