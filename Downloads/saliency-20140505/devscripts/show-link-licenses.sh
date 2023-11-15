#!/bin/sh

# $Id: show-link-licenses.sh 7898 2007-02-12 21:41:46Z rjpeters $
# $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/show-link-licenses.sh $

# Use the rpm database to show the licenses of libraries against which
# a given program is linked; usage: "show-link-licenses.sh bin/ezvision".

for f in "$@"; do
    deps=`ldd $f | awk '{print $3}'`
    for dep in $deps; do
	if test -f $dep; then
	    pkg=`rpm -qf $dep`
	    if test $? -eq 0; then
		base=`basename $dep`
		license=`rpm -qi $pkg | grep -o 'License:.*'`
		printf "%30s (%20s) $license\n" $pkg $base
	    fi
	fi
    done
done
