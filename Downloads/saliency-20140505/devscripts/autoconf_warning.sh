#!/bin/sh

# $Id: autoconf_warning.sh 4294 2005-06-04 05:58:57Z rjpeters $

# This is a helper script for sorting through the various snafus that can
# arise in autoconf/configure/configure.ac handling, due to the various
# ways that cvs handles timestamps.

config_ac_revision=`grep configure.ac CVS/Entries | cut -d '/' -f 3`
config_ac_checkout_time=`grep configure.ac CVS/Entries | cut -d '/' -f 4`
config_ac_modify_time=`date -r configure.ac`

# Put both time-date strings into the same format. We have to trick 'date'
# into ignoring timezones by using the '-u' option, because otherwise the
# modification-time of configure.ac will appear to be in the local
# timezone, while the item from CVS/Entries will appear to be in UTC0 time
# (even though it is actually a local timestamp).

config_ac_checkout_time=`date -u -d "$config_ac_checkout_time"`
config_ac_modify_time=`date -u -d "$config_ac_modify_time"`

echo "'configure.ac' revision is $config_ac_revision"
echo "'configure.ac' checkout time is     '$config_ac_checkout_time'"
echo "'configure.ac' modification time is '$config_ac_modify_time'"

if test "$config_ac_checkout_time" = "$config_ac_modify_time"; then
    echo "'configure.ac' appears to be unmodified"

    config_src_revision=`grep "From configure\.ac Revision" configure | cut -d ' ' -f 5`

    echo "'configure' was generated from configure.ac revision $config_src_revision"

    if test $config_src_revision = $config_ac_revision; then
	echo "'configure' appears to be current relative to configure.ac"
	echo "Timestamps will be updated accordingly."
	touch configure
	exit 0
    fi
fi

echo "WARNING: your 'configure' script appears to be out-of-date relative to"
echo "'configure.ac'. However, the configure script could not find an 'autoconf'"
echo "executable with version >= 2.53 with which to regenerate 'configure'."
echo "There are three possibilities:"
echo "  (1) 'configure' is actually up-to-date, but the timestamps don't reflect"
echo "      this (because of a cvs update, for example). In this case, you can"
echo "      just 'touch configure' to update its timestamp."
echo "  (2) You have modified 'configure.ac'. If you don't have autoconf >= 2.53"
echo "      installed, then you are stuck in this case since you don't have a way"
echo "      to update the 'configure' script. You should revert back to a clean"
echo "      copy of 'configure.ac'."
echo "  (3) You do have autoconf >= 2.53 installed somewhere, but the configure"
echo "      script was unable to find it. You can go into the Makefile and"
echo "      hand-edit the AUTOCONF_PROG variable to point to the right program."
echo ""
exit 1
