#!/bin/sh

# $Id: query-svn-unmodified.sh 5462 2005-09-07 19:15:08Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/query-svn-unmodified.sh $

# Do "svn status" on all files in argv; if all files are (1) known to
# svn and (2) not locally modified, then exit with a status of 0;
# otherwise exit with a status of 1. This provides a quick way for the
# makefile to check if a batch of files are locally unmodified.

for f in "$@"; do
    if test -f "$f"; then

        # ok, the file exists, now let's check its svn status

        stat=`svn status $f`
        code=$?

        if test $code -ne 0; then
            echo "svn status '$f' failed with exit code $code"
            exit 1
        fi

        if test "x$stat" = "x"; then
            # ok, that file was ok, so let's go on to the next one
            echo "'$f' is locally unmodified"
            continue
        else

            echo "'$f' is LOCALLY MODIFIED (status='$stat')"
            exit 1
        fi

    else

        echo "no such file: '$f'"
        exit 1
    fi
done

exit 0
