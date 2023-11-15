dnl Process this file with autoconf to produce a configure script.

##########################################################################
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the ##
## University of Southern California (USC) and the iLab at USC.         ##
## See http://iLab.usc.edu for information about this project.          ##
##########################################################################
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; filed July 23, 2001, following provisional applications     ##
## No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).##
##########################################################################
## This file is part of the iLab Neuromorphic Vision C++ Toolkit.       ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is free software; you can   ##
## redistribute it and/or modify it under the terms of the GNU General  ##
## Public License as published by the Free Software Foundation; either  ##
## version 2 of the License, or (at your option) any later version.     ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  ##
## that it will be useful, but WITHOUT ANY WARRANTY; without even the   ##
## implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ##
## PURPOSE.  See the GNU General Public License for more details.       ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   ##
## to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   ##
## Boston, MA 02111-1307 USA.                                           ##
##########################################################################
##
## Primary maintainer for this file: Rob Peters <rjpeters@klab.caltech.edu>
## $Id: configure.ac 15493 2014-01-22 21:38:02Z itti $
##

AC_INIT([iLab C++ Neuromorphic Vision Toolkit], [3.1], [http://ilab.usc.edu/forum/], [ilabkit])
AC_PREREQ(2.53)

AC_REVISION($Revision: 15493 $)

# Note, unfortunately we can't do AC_PREFIX_DEFAULT(`pwd`) here,
# because then we can no longer run simple make commands like 'make
# bin/ezvision'; instead we have to do 'make
# /full/long/absolute/path/to/saliency/bin/ezvision'. So, we're stuck
# with AC_PREFIX_DEFAULT(.).
AC_PREFIX_DEFAULT(.)

# Have a look here for lots of macros:
# http://www.gnu.org/software/autoconf-archive/The-Macros.html
# these get installed via urpmi autoconf-archive and get used by aclocal

##########################################################################
### New macro definitions

dnl AC_CHECK_LIB_CXX(libname, includes, function-call,
dnl                  action-on-success, action-on-fail,
dnl                  extra-libs-for-link)

AC_DEFUN([AC_CHECK_LIB_CXX],
         [AC_MSG_CHECKING([for lib$1])
         libs_save=$LIBS
         LIBS="$LIBS -l$1 $6"
         AC_TRY_LINK([$2], [$3], [havelib_$1=yes], [havelib_$1=no])
         LIBS=$libs_save
         if test "x$havelib_$1" = xyes; then
            AC_MSG_RESULT(yes)
            $4
         else
            AC_MSG_RESULT(no)
            $5
         fi])


dnl AC_CHECK_LIB_CXX_FULL(1=libname, 2=macro-name, 3=libs-var-name,
dnl                  4=extra-warning-message,
dnl                  5=includes, 6=function-call,
dnl                  7=action-on-success, 8=action-on-fail,
dnl                  9=extra-libs-for-link)

AC_DEFUN([AC_CHECK_LIB_CXX_FULL],
         [AC_MSG_CHECKING([for lib$1])
         m4_define([MOD1], [m4_translit([$1], [-], [_])])
         AH_TEMPLATE([$2],1,[have lib$1 library and associated headers?])
         libs_save=$LIBS
         LIBS="$LIBS -l$1 $9"
         AC_TRY_LINK([$5], [$6], [havelib_MOD1=yes], [havelib_MOD1=no])
         LIBS=$libs_save
         if test "x$havelib_MOD1" = xyes; then
            AC_DEFINE([$2],1,[have lib$1 library and associated headers?])
            $3="-l$1 $9"
            $2=yes
            AC_MSG_RESULT([yes ($3="-l$1 $9" and $2=yes)])
            AC_FEATURE_STATUS([lib$1], [ok])
            $7
            AC_SUBST(with_[]MOD1, [yes])
         else
            $3=""
            $2=no
            AC_MSG_RESULT(no)
            AC_FEATURE_STATUS([lib$1], [missing], [$4])
            AC_SUBST(with_[]MOD1, [no])
            $8
         fi
         AC_SUBST($3)])

dnl AC_CHECK_ACLOCAL_MACRO(1=macroname, 2=packagename, 3=extra-warning-message)
dnl This runs macroname (e.g., this can be things like AX_BOOST_BASE), and reports
dnl on status using packagename and possible extra warnings. The macro should
dnl report $succeeded=yes when successful.
AC_DEFUN([AC_CHECK_ACLOCAL_MACRO],
	[$1
	if test "$succeeded" = "yes" ; then AC_FEATURE_STATUS([$2], [ok])
	else AC_FEATURE_STATUS([$2], [missing], [$3]); fi
	])

dnl AC_CHECK_LIB_CXX_ENABLED(1=libname, 2=macro-name, 3=libs-var-name,
dnl                  4=extra-warning-message,
dnl                  5=includes, 6=function-call,
dnl                  7=action-on-success, 8=action-on-fail,
dnl                  9=extra-libs-for-link)
dnl This is like AC_CHECK_CXX_FULL plus we create a configure command-line
dnl option to enable/disable this feature. By default, the feature is enabled,
dnl and the command-line option is to disable it.
AC_DEFUN([AC_CHECK_LIB_CXX_ENABLED],
         [AC_MSG_CHECKING([whether to enable support for $1])
          AC_ARG_ENABLE($1, [AC_HELP_STRING([--disable-$1], [disable support for $1])],
                        [enable_it=$enableval], [enable_it=yes])
          AC_MSG_RESULT($enable_it)
          if test "x$enable_it" = xyes; then
             AC_CHECK_LIB_CXX_FULL([$1],[$2],[$3],[$4],[$5],[$6],[$7],[$8],[$9])
          else
             AC_FEATURE_STATUS([lib$1], [disabled], [(disabled by ./configure --disable-$1)])
             AC_SUBST([with_$1], [no])
          fi])

dnl AC_CHECK_LIB_CXX_DISABLED(1=libname, 2=macro-name, 3=libs-var-name,
dnl                  4=extra-warning-message,
dnl                  5=includes, 6=function-call,
dnl                  7=action-on-success, 8=action-on-fail,
dnl                  9=extra-libs-for-link)
dnl This is like AC_CHECK_CXX_FULL plus we create a configure command-line
dnl option to enable/disable this feature. By default, the feature is disabled,
dnl and the command-line option is to enable it.
AC_DEFUN([AC_CHECK_LIB_CXX_DISABLED],
         [AC_MSG_CHECKING([whether to enable support for $1])
          AC_ARG_ENABLE($1, [AC_HELP_STRING([--enable-$1], [enable support for $1])],
                        [enable_it=$enableval], [enable_it=no])
          AC_MSG_RESULT($enable_it)
          if test "x$enable_it" = xyes; then
             AC_CHECK_LIB_CXX_FULL([$1],[$2],[$3],[$4],[$5],[$6],[$7],[$8],[$9])
          else
             AC_FEATURE_STATUS([lib$1], [disabled], [(use ./configure --enable-$1 to enable)])
             AC_SUBST([with_$1], [no])
          fi])



dnl AC_DEFINE_IF_COMPILE(macroname, description, includes, testcode)
dnl
dnl Use this macro to AC_DEFINE a symbol only if a given piece of code
dnl can be compiled successfully. Also prints an AC_MSG_CHECKING
dnl message, and a yes/no result.

AC_DEFUN([AC_DEFINE_IF_COMPILE],
         [AH_TEMPLATE([$1], [$2])
          AC_MSG_CHECKING([$2])
          AC_COMPILE_IFELSE([AC_LANG_PROGRAM([$3], [$4])],
                            [AC_MSG_RESULT([yes])
                             AC_DEFINE([$1],1,[$2])],
                            [AC_MSG_RESULT([no])])])

dnl AC_DEFINE_IF_NOT_COMPILE(macroname, description, includes, testcode)
dnl
dnl Use this macro to AC_DEFINE a symbol only if a given piece of code
dnl CANNOT be compiled successfully. Also prints an AC_MSG_CHECKING
dnl message, and a yes/no result. Make sure that your AC_MSG_CHECKING
dnl description reflects the inverted sense of this test; e.g.,
dnl something like 'whether SYMBOL_FOO is missing'.

AC_DEFUN([AC_DEFINE_IF_NOT_COMPILE],
         [AH_TEMPLATE([$1], [$2])
          AC_MSG_CHECKING([$2])
          AC_COMPILE_IFELSE([AC_LANG_PROGRAM([$3], [$4])],
                            [AC_MSG_RESULT([no])],
                            [AC_MSG_RESULT([yes])
                             AC_DEFINE([$1],1,[$2])])])

dnl AC_DEFINE_IF_LINK(macroname, description, includes, testcode)
dnl
dnl Use this macro to AC_DEFINE a symbol only if a given piece of code
dnl compiles and links successfully. Also prints an AC_MSG_CHECKING
dnl message, and a yes/no result.

AC_DEFUN([AC_DEFINE_IF_LINK],
         [AH_TEMPLATE([$1], [$2])
          AC_MSG_CHECKING([$2])
          AC_LINK_IFELSE([AC_LANG_PROGRAM([$3], [$4])],
                         [AC_MSG_RESULT([yes])
                          AC_DEFINE([$1],1,[$2])],
                         [AC_MSG_RESULT([no])])])

dnl AC_DEFINE_IF_LINK(macroname, description, includes, testcode)
dnl
dnl Use this macro to AC_DEFINE a symbol only if a given piece of code
dnl DOES NOT compile and link successfully. Also prints an
dnl AC_MSG_CHECKING message, and a yes/no result. Make sure that your
dnl AC_MSG_CHECKING description reflects the inverted sense of this
dnl test; e.g., something like 'whether SYMBOL_FOO is missing'.

AC_DEFUN([AC_DEFINE_IF_NOT_LINK],
         [AH_TEMPLATE([$1], [$2])
          AC_MSG_CHECKING([$2])
          AC_LINK_IFELSE([AC_LANG_PROGRAM([$3], [$4])],
                         [AC_MSG_RESULT([no])],
                         [AC_MSG_RESULT([yes])
                          AC_DEFINE([$1],1,[$2])])])

dnl AC_DEFINE_IF_RUN(macroname, description, includes, testcode)
dnl
dnl Use this macro to AC_DEFINE a symbol only if a given piece of code
dnl compiles, links and exits with a 0 exit status. Also prints an
dnl AC_MSG_CHECKING message, and a yes/no result.

AC_DEFUN([AC_DEFINE_IF_RUN],
         [AH_TEMPLATE([$1], [$2])
          AC_MSG_CHECKING([$2])
          AC_RUN_IFELSE([AC_LANG_PROGRAM([$3], [$4])],
                        [AC_MSG_RESULT([yes])
                         AC_DEFINE([$1],1,[$2])],
                        [AC_MSG_RESULT([no])])])

dnl AC_DEFINE_IF_RUN(macroname, description, includes, testcode)
dnl
dnl Use this macro to AC_DEFINE a symbol only if a given piece of code
dnl DOES NOT compile, link, and exit with a 0 exit status. Also prints
dnl an AC_MSG_CHECKING message, and a yes/no result. Make sure that
dnl your AC_MSG_CHECKING description reflects the inverted sense of
dnl this test; e.g., something like 'whether SYMBOL_FOO is missing'.

AC_DEFUN([AC_DEFINE_IF_NOT_RUN],
         [AH_TEMPLATE([$1], [$2])
          AC_MSG_CHECKING([$2])
          AC_RUN_IFELSE([AC_LANG_PROGRAM([$3], [$4])],
                        [AC_MSG_RESULT([no])],
                        [AC_MSG_RESULT([yes])
                         AC_DEFINE([$1],1,[$2])])])

dnl AC_LATE_WARN(msg) msg will be given as a warning, and will be
dnl repeated at the end of running the configure script, making it more
dnl likely that the user will notice it.

AC_DEFUN([AC_LATE_WARN],
[AC_MSG_WARN([$1])
late_warnings="${late_warnings}
* $1"])

# AC_GIVE_LATE_WARNINGS() actually print the late warnings
AC_DEFUN([AC_GIVE_LATE_WARNINGS],
[if test "x$late_warnings" != "x"; then
   AC_MSG_NOTICE([WARNINGS:$late_warnings])
fi])

late_warnings=""

dnl AC_FEATURE_STATUS(featurename, status, msg) will note the status
dnl of an optional feature (such as external library dependency) so
dnl that it can be reported later with AC_REPORT_FEATURE_STATUS
AC_DEFUN([AC_FEATURE_STATUS],
[invt_lib_status_featurename="AS_ESCAPE([$1], [\$"`])"
invt_lib_status_status="AS_ESCAPE([$2], [\$"`])"
invt_lib_status_msg="AS_ESCAPE([$3], [\$"`])"
invt_lib_status_line=`printf '%-25s   %-8s   %s' "$invt_lib_status_featurename" "$invt_lib_status_status" "$invt_lib_status_msg"`
invt_lib_status="${invt_lib_status}
* $invt_lib_status_line"])
dnl "  # keep emacs happy

dnl AC_REPORT_FEATURE_STATUS() report what was recorded with
dnl AC_FEATURE_STATUS()
AC_DEFUN([AC_REPORT_FEATURE_STATUS],
[if test "x$invt_lib_status" != "x"; then
    AC_MSG_NOTICE([status of optional features:$invt_lib_status])
fi])

dnl AC_DEFINE_DIR(VARNAME, DIR [, DESCRIPTION])
dnl modified from http://autoconf-archive.cryp.to/ac_define_dir.html
AC_DEFUN([AC_DEFINE_DIR], [
    prefix_NONE=
    exec_prefix_NONE=
    prefix_DOT=
    if test "x$prefix" = xNONE; then
        prefix_NONE=yes
        prefix=$ac_default_prefix
    fi
    if test "x$prefix" = "x."; then
        prefix_DOT=yes
        prefix=`pwd`
    fi
    if test "x$exec_prefix" = xNONE; then
        exec_prefix_NONE=yes
        exec_prefix=$prefix
    fi
dnl In Autoconf 2.60, ${datadir} refers to ${datarootdir}, which in turn
dnl refers to ${prefix}.  Thus we have to use `eval' twice.
    eval ac_define_dir="\"[$]$2\""
    eval ac_define_dir="\"$ac_define_dir\""
    AC_SUBST($1, "$ac_define_dir")
    AC_DEFINE_UNQUOTED($1, "$ac_define_dir", [$3])
    if test "$prefix_NONE"; then
        prefix=NONE
    fi
    if test "$prefix_DOT"; then
        prefix=.
    fi
    if test "$exec_prefix_NONE"; then
        exec_prefix=NONE
    fi
])

##########################################################################
### Figure out where we're building from

source="src/"
objdir="build/obj/"
buildscripts="./devscripts"

AC_MSG_CHECKING([for source directory])
AC_MSG_RESULT($source)

AC_MSG_CHECKING([for object file directory])
AC_MSG_RESULT($objdir)

AC_MSG_CHECKING([for build scripts directory])
AC_MSG_RESULT($buildscripts)

AC_SUBST(source)
AC_SUBST(objdir)
AC_SUBST(buildscripts)

AC_DEFINE_DIR([INVT_PLUGIN_DIR], [exec_prefix/lib/invt/plugins],
              [where plugin libraries are installed])

AC_MSG_CHECKING([for plugin directory])
AC_MSG_RESULT($INVT_PLUGIN_DIR)

AC_DEFINE_DIR([INVT_LIB_DIR], [exec_prefix/lib/invt],
              [where auxiliary runtime-library files are installed])

AC_MSG_CHECKING([for runtime-library directory])
AC_MSG_RESULT($INVT_LIB_DIR)

##########################################################################
### Determine machine architecture and capabilities

uname_s=unknown
uname_p=unknown
if test -x /bin/uname; then
   uname_s=`/bin/uname -s`
   uname_p=`/bin/uname -p`
elif test -x /usr/bin/uname; then
   uname_s=`/usr/bin/uname -s`
   uname_p=`/usr/bin/uname -p`
fi

AC_MSG_CHECKING([for OS])
AC_MSG_RESULT($uname_s)

AC_MSG_CHECKING([for Darwin (MacOS-X) system?])
if test "x$uname_s" = xDarwin; then
  AC_DEFINE([MACHINE_OS_DARWIN], 1, [Machine's O.S. is Darwin (MacOS-X)])
  AC_MSG_RESULT([yes])
else
  AC_MSG_RESULT([no])
fi

AC_MSG_CHECKING([for CPU type])
AC_MSG_RESULT($uname_p)

cpu_supports_mmx=no
cpu_supports_sse=no
cpu_supports_sse2=no
cpu_supports_sse3=no
cpu_supports_ssse3=no
cpu_supports_sse41=no
cpu_supports_3dnow=no
cpu_supports_64bit=no
cpu_model=""
if test -r /proc/cpuinfo; then
    cpu_info="cat /proc/cpuinfo";
elif test -r /compat/linux/proc/cpuinfo; then
    cpu_info="cat /compat/linux/proc/cpuinfo"
elif test -r /usr/sbin/sysctl; then
    #perhaps we have a freeBsSD machine, then we can use sysctl	
    cpu_info="/usr/sbin/sysctl -a machdep"
    cpu_machdep=yes		
else
    # todo: write a program to get cpu type
    # in the meantime, try something that will fail
    cpu_info="/bin/where_is_cpu_info"
fi

# The Cell Broadband Engine CPU, for example here in a Sony
# PlayStation 3, exhibits a significantly peculiar /proc/cpuinfo which
# deserves to be treated separately:
#
#processor       : 0
#cpu             : Cell Broadband Engine, altivec supported
#clock           : 3192.000000MHz
#revision        : 5.1 (pvr 0070 0501)
#
#processor       : 1
#cpu             : Cell Broadband Engine, altivec supported
#clock           : 3192.000000MHz
#revision        : 5.1 (pvr 0070 0501)
#
#timebase        : 79800000
#machine         : PS3PF

cpu_machine=`$cpu_info| grep 'machine' | cut -d ':' -f 2|head -1`
if test "x$cpu_machine" = "x PS3PF"; then
   cpu_model=cell
   cpu_is_x86_compatible=no
   cpu_supports_64bit=yes
   cpu_flags=""
   cpu_vendor="STI (Sony/Toshiba/IBM)"
   cpu_family=0
   cpu_t_model=0
elif test "x$cpu_machdep" = "xyes"; then
#we have a machdep from sysctl which has most of the 
#info from /proc/cpuinfo but in a different format:
  cpu_flags=`$cpu_info | grep 'cpu.features' | cut -d ':' -f 2 | head -1 |
              tr [A-Z] [a-z]`
  cpu_vendor=`$cpu_info | grep cpu.vendor | cut -d ':' -f 2 | \
              cut -d ' ' -f 2 | head -1`
  cpu_family=`$cpu_info | grep 'cpu.family' | cut -d ':' -f 2 | \
              cut -d ' ' -f 2 | head -1`
  cpu_t_model=`$cpu_info | grep 'cpu.model' | cut -d ':' -f 2 | \
	      cut -d ' ' -f 2 | head -1`
else
   # These work with more conventional machines:
   cpu_flags=`$cpu_info | grep 'flags' | cut -d ':' -f 2 | head -1`
   cpu_vendor=`$cpu_info | grep vendor_id | cut -d ':' -f 2 | \
              cut -d ' ' -f 2 | head -1`
   cpu_family=`$cpu_info| grep 'cpu family' | cut -d ':' -f 2 | \
              cut -d ' ' -f 2 | head -1`
   cpu_t_model=`$cpu_info|grep -v 'model name' | grep 'model' | \
              cut -d ':' -f 2 |cut -d ' ' -f 2 | head -1`
fi

# check for mmx, sse, etc and for the 'lm' (long mode) flag which
# indicates a 64-bit CPU:
for i in $cpu_flags; do
    case "$i" in
        mmx)      cpu_supports_mmx=yes            ;;
        sse)      cpu_supports_sse=yes            ;;
        sse2)     cpu_supports_sse2=yes            ;;
        sse3)     cpu_supports_sse3=yes     ;;
        ssse3)    cpu_supports_ssse3=yes    ;;
        sse41)    cpu_supports_sse41=yes    ;;
        3dnow)    cpu_supports_3dnow=yes    ;;
        mmxext)   cpu_supports_mmxext=yes   ;;
        3dnowext) cpu_supports_3dnowext=yes ;;
        lm)       cpu_supports_64bit=yes    ;;
    esac
done

cpu_is_x86_compatible=no

for vendor  in $cpu_vendor; do
    case $vendor in
    GenuineIntel)
        case "$cpu_family" in
        3) cpu_model=i386 ;;
        4) cpu_model=i486 ;;
        5) if test "$cpu_t_pmodel" -eq 4 -o "$pmodel" -eq 8; then
                cpu_model=pentium-mmx
            else
                cpu_model=i586
            fi
            ;;
        6) if test "$cpu_t_model" -ge 15; then
	      	cpu_model=core2 
	    elif test "$cpu_t_model" -ge 7; then
                cpu_model=pentium3
            elif test "$cpu_t_model" -ge 3; then
                cpu_model=pentium2
            else
                cpu_model=i686
            fi
            ;;
        15) cpu_model=pentium4 ;;
        *) cpu_model=pentium4  ;;
        esac
        cpu_is_x86_compatible=yes
        ;;
    AuthenticAMD)
        case "$cpu_family" in
        3) cpu_model=i386 ;;
        4) cpu_model=i486  ;;
        5)
            if test "$cpu_t_model" -eq 9; then
                cpu_model=k6-3
            elif test "$cpu_t_model" -ge 8; then
                cpu_model=k6-2
            elif test "$cpu_t_model" -ge 6; then
                cpu_model=k6
            else
                cpu_model=i586
            fi
            ;;
        6)
            if test "$cpu_t_model" -ge 7; then
                cpu_model=athlon-4
            elif test "$cpu_t_model" -ge 6; then
                cpu_model=athlon-xp
            elif test "$cpu_t_model" -ge 5; then
                cpu_model=opteron
            elif test "$cpu_t_model" -ge 4; then
                cpu_model=athlon-tbird
            else
                cpu_model=athlon
            fi
            ;;
        15) cpu_model=opteron
            AC_DEFINE(INVT_CPU_OPTERON,1,[have an AMD Opteron cpu])
            ;;
        esac
        cpu_is_x86_compatible=yes
        ;;
    unknown)
        case "$cpu_family" in
        3) cpu_model=i386 ;;
        4) cpu_model=i486 ;;
        5) cpu_model=i586 ;;
        *) cpu_model=i686 ;;
            esac
        ;;
    esac
done

AC_MSG_CHECKING([for CPU model])
AC_MSG_RESULT($cpu_model)

AC_MSG_CHECKING([whether $cpu_model CPU is x86-compatible])
AC_MSG_RESULT($cpu_is_x86_compatible)

if test "x$cpu_is_x86_compatible" = "xyes"; then
   AC_DEFINE(INVT_CPU_IX86,1,[have an x86(-compatible) cpu])
fi

AC_MSG_CHECKING([whether $cpu_model CPU supports Intel MMX instructions])
AC_MSG_RESULT($cpu_supports_mmx)

AC_MSG_CHECKING([whether $cpu_model CPU supports Intel SSE instructions])
AC_MSG_RESULT($cpu_supports_sse)

AC_MSG_CHECKING([whether $cpu_model CPU supports Intel SSE2 instructions])
AC_MSG_RESULT($cpu_supports_sse2)

AC_MSG_CHECKING([whether $cpu_model CPU supports Intel SSE3 instructions])
AC_MSG_RESULT($cpu_supports_sse3)

AC_MSG_CHECKING([whether $cpu_model CPU supports Intel SSSE3 instructions])
AC_MSG_RESULT($cpu_supports_ssse3)

AC_MSG_CHECKING([whether $cpu_model CPU supports Intel SSE41 instructions])
AC_MSG_RESULT($cpu_supports_sse41)

AC_MSG_CHECKING([whether $cpu_model CPU supports AMD 3Dnow instructions])
AC_MSG_RESULT($cpu_supports_3dnow)

AC_MSG_CHECKING([whether $cpu_model CPU supports 64 bit instructions])
AC_MSG_RESULT($cpu_supports_64bit)

##########################################################################
# bitsfx is a suffix that will be set to either empty for a 32-bit
# build, or "64" for a 64-bit build. Below we will test whether the
# cpu supports 64-bit and the user has not selected --enable-force32:

# should we force 32 bit code?

AC_MSG_CHECKING([whether to force 32 bit code under 64 bit CPU])

AC_ARG_ENABLE(force32,
              [AC_HELP_STRING([--enable-force32],
                             [force the compiler to compile 32 bit code])],
              ,
              enable_force32=no)

AC_MSG_RESULT($enable_force32)

if test "x$enable_force32" = xyes; then
   AC_DEFINE(INVT_USE_32_BIT,1,[Are we using 32 bit code?])
elif test "x$cpu_supports_64bit" = xyes; then
   AC_DEFINE(INVT_USE_64_BIT,1,[Are we using 64 bit code?])
fi

bitsfx=""

if test "x$cpu_supports_64bit" = xyes; then
   # ok, do a 64-bit build unless user does not want it by specifying
   # --enable-force32:
   if test "x$enable_force32" = xno; then bitsfx="64"; fi
fi

# below, use bitsfx wjen specifying things like lib paths, etc

##########################################################################
### Process command-line options and other user settings

# If CXXFLAGS was not already set, then set it now to keep autoconf from
# adding "-g -O2".

if test "${CXXFLAGS+set}" != "set" ; then
    CXXFLAGS=""
fi

if test "${CFLAGS+set}" != "set" ; then
    CFLAGS=""
fi

# Q: Why do we have a CXXWARNINGS variable, separate from CXXFLAGS?

# A: Because if CXXFLAGS contains e.g. "-Wall -Werror" during the
# execution of this configure script, then some of the configuration
# tests might fail because they generate warnings, even though we
# don't really want to treat warnings as errors *during the configure
# script*. In particular, the tests for finding link libraries tend to
# generate warnings, because the way those tests work is to compile a
# trivial function call to one of the functions in the library, and
# see if the link succeeds or fails; but, since we don't do anything
# with the result of that function call, we may get a "warning: this
# statement has no effect", which becomes an error if we have
# -Werror. Therefore: we set up CXXWARNINGS separately, and add it to
# CXXFLAGS only at the very end of the configure script.

CXXWARNINGS=""
CWARNINGS=""

AC_DEFINE(_GNU_SOURCE,1,[allow GNU libc extensions?])
AC_DEFINE(_REENTRANT,1,[allow for reentrant code?])
AC_DEFINE(INVT_INST_BYTE,1,[instantiate templates for T=byte?])
AC_DEFINE(INVT_INST_FLOAT,1,[instantiate templates for T=float?])

### --with-lib64

AC_MSG_CHECKING([whether to use a lib64 directory])

AC_ARG_WITH(lib64,
            [AC_HELP_STRING([--with-lib64=DIR],
                            [lib64 directory to search for 64-bit libraries [default=/usr/lib64]])],
            ,
            with_lib64="/usr/lib64")

if test "x$with_lib64" = "xno" -o "x$with_lib64" = "x"; then
   AC_MSG_RESULT([no])
   with_lib64=""
elif test -d "$with_lib64"; then
   LDFLAGS="$LDFLAGS -L$with_lib64"
   AC_MSG_RESULT([$with_lib64])
fi

### --with-tcl, --with-tk

AC_ARG_WITH(tcl,
            [AC_HELP_STRING([--with-tcl=DIR],
                            [directory containing tclConfig.sh from tcl >= 8.4 [default=/usr/lib]])],
            ,
            with_tcl="")

if test "x$with_tcl" = x; then
   if test -d "$with_lib64"; then
      with_tcl=$with_lib64
   else
      with_tcl=/usr/lib
   fi
fi

AC_ARG_WITH(tk,
            [AC_HELP_STRING([--with-tk=DIR],
                            [directory containing tkConfig.sh from tk >= 8.4 [default=--with-tcl]])],
            ,
            with_tk="")

if test "x$with_tk" = x; then
   with_tk=$with_tcl
fi

AC_MSG_CHECKING([for location of tcl >= 8.4])

if test -r "$with_tcl/tclConfig.sh"; then
   :
elif test -r "$with_tcl/tcl8.4/tclConfig.sh"; then
   with_tcl="$with_tcl/tcl8.4"
elif test -r "/usr/lib/tcl8.4/tclConfig.sh"; then
   with_tcl="/usr/lib/tcl8.4"
fi

AC_MSG_RESULT($with_tcl)

AC_MSG_CHECKING([for location of tk >= 8.4])

if test -r "$with_tk/tkConfig.sh"; then
   :
elif test -r "$with_tk/tk8.4/tkConfig.sh"; then
   with_tk="$with_tk/tk8.4"
elif test -r "/usr/lib/tk8.4/tkConfig.sh"; then
   with_tk="/usr/lib/tk8.4"
fi

AC_MSG_RESULT($with_tk)




### --enable-mmx

AC_MSG_CHECKING([whether to enable Intel MMX optimizations])

AC_ARG_ENABLE(mmx,
              [AC_HELP_STRING([--enable-mmx],
                              [enable Intel MMX optimizations])],
              ,
              enable_mmx=no)

if test "x$enable_mmx" = xyes; then
   AC_DEFINE(INVT_USE_MMX,1,[enable Intel MMX optimizations?])
fi

AC_MSG_RESULT($enable_mmx)



### --enable-sse

AC_MSG_CHECKING([whether to enable Intel SSE optimizations])

AC_ARG_ENABLE(sse,
              [AC_HELP_STRING([--enable-sse],
                              [enable Intel SSE optimizations (Pentium-III, Celeron, Athlon)])],
              ,
              enable_sse=no)

if test "x$enable_sse" = xyes; then
   AC_DEFINE(INVT_USE_SSE,1,[enable Intel SSE optimizations?])
fi

AC_MSG_RESULT($enable_sse)



### --enable-sse2

AC_MSG_CHECKING([whether to enable Intel SSE2 optimizations])

AC_ARG_ENABLE(sse2,
              [AC_HELP_STRING([--enable-sse2],
                              [enable Intel SSE2 optimizations (Pentium-4, Celeron-II)])],
              ,
              enable_sse2=no)

if test "x$enable_sse2" = xyes; then
   AC_DEFINE(INVT_USE_SSE2,1,[enable Intel SSE2 optimizations?])
fi

AC_MSG_RESULT($enable_sse2)

### --enable-sse3

AC_MSG_CHECKING([whether to enable Intel SSE3 optimizations])

AC_ARG_ENABLE(sse3,
              [AC_HELP_STRING([--enable-sse3],
                              [enable Intel SSE3 optimizations (Pentium-4, Celeron-II)])],
              ,
              enable_sse3=no)

if test "x$enable_sse3" = xyes; then
   AC_DEFINE(INVT_USE_SSE3,1,[enable Intel SSE3 optimizations?])
   CXXFLAGS="$CXXFLAGS -msse3"
   CFLAGS="$CFLAGS -msse3"
fi

AC_MSG_RESULT($enable_sse3)

### --enable-sse-debayer

AC_MSG_CHECKING([whether to enable Intel SSE/2/3 debayer optimizations])

AC_ARG_ENABLE(ssedb,
              [AC_HELP_STRING([--enable-ssedb],
                              [enable Intel SSE/2/3 optimizations for de-Bayer code])],
              ,
              enable_ssedb=no)

if test "x$enable_ssedb" = xyes; then
   AC_DEFINE(INVT_USE_SSEDB,1,[enable Intel SSE/2/3 optimizations for de-Bayer code?])
   CXXFLAGS="$CXXFLAGS -msse3"
   CFLAGS="$CFLAGS -msse3"
fi

AC_MSG_RESULT($enable_ssedb)


### --enable-aggressive

AC_MSG_CHECKING([whether to use cpu-tailored aggressive optimizations])

AC_ARG_ENABLE(aggressive,
              [AC_HELP_STRING([--enable-aggressive],
                              [use aggressive optimizations tailored to cpu model])],
              ,
              enable_aggressive=no)

AC_MSG_RESULT($enable_aggressive)

if test "x$enable_aggressive" = xyes; then
   AC_LATE_WARN([you have selected aggressive optimizations; binaries will be less portable])
fi


### --enable-ffast-math
# check to use fast math optimizations

AC_MSG_CHECKING([whether to use non-IEEE non-ISO fast math optimization])

AC_ARG_ENABLE(ffastmath,
              [AC_HELP_STRING([--enable-ffastmath],
                             [use non-IEEE non-ISO fast math optimization])],
              ,
              enable_ffastmath=no)

AC_MSG_RESULT($enable_ffastmath)

if test "x$enable_ffastmath" = xyes; then
   AC_LATE_WARN([you have selected fast math optimizations which are non-IEEE non-ISO compliant])
   CXXFLAGS="$CXXFLAGS -ffast-math"
   CFLAGS="$CFLAGS -ffast-math"
fi


### --enable-profilearcs
# should we use profile arcs to discover optimizations at run time (GCC 3.3ish)

AC_MSG_CHECKING([whether to use arc profiles for optimizations])

AC_ARG_ENABLE(profilearcs,
              [AC_HELP_STRING([--enable-profilearcs],
                             [create arc profiles for run time optimizations GCC 3.3])],
              ,
              enable_profilearcs=no)

AC_MSG_RESULT($enable_profilearcs)

if test "x$enable_profilearcs" = xyes; then
   AC_LATE_WARN([you have selected to use profile arcs If you dont userstand what this is dont use it])
   CXXFLAGS="$CXXFLAGS -fprofile-arcs"
   CFLAGS="$CFLAGS -fprofile-arcs"
fi


### --enable-profilegen
# should we use profile generation to discover optimizations at run time (GCC 3.4+)

AC_MSG_CHECKING([whether to use profile generate for optimizations])

AC_ARG_ENABLE(profilegen,
              [AC_HELP_STRING([--enable-profilegen],
                             [generate profiles for run time optimizations GCC 3.4])],
              ,
              enable_profilegen=no)

AC_MSG_RESULT($enable_profilegen)

if test "x$enable_profilegen" = xyes; then
   AC_LATE_WARN([you have selected to use profile generate If you dont userstand what this is dont use it])
   CXXFLAGS="$CXXFLAGS -fprofile-generate"
   CFLAGS="$CFLAGS -fprofile-generate"
fi


### --enable-fbranch
# should we use arc profile to compute branch probabilities

AC_MSG_CHECKING([whether to compute branch probabilities])

AC_ARG_ENABLE(fbranch,
              [AC_HELP_STRING([--enable-fbranch],
                             [create branch probabilites from arc profiles GCC 3.3])],
              ,
              enable_fbranch=no)

AC_MSG_RESULT($enable_fbranch)

if test "x$enable_fbranch" = xyes; then
   AC_LATE_WARN([you have selected to use branch probabilities You must first use profile-arc])
   CXXFLAGS="$CXXFLAGS -fbranch-probabilities"
   CFLAGS="$CFLAGS -fbranch-probabilities"
fi


### --enable-fprofile
# should we use da profile to compute branch probabilities

AC_MSG_CHECKING([whether to use branch profiles])

AC_ARG_ENABLE(fprofile,
              [AC_HELP_STRING([--enable-fprofile],
                             [create branch probabilites from profiles GCC 3.4])],
              ,
              enable_fprofile=no)

AC_MSG_RESULT($enable_fprofile)

if test "x$enable_fprofile" = xyes; then
   AC_LATE_WARN([you have selected to use branch probabilities You must first use profile generate])
   CXXFLAGS="$CXXFLAGS -fprofile-use"
   CFLAGS="$CFLAGS -fprofile-use"
fi


### --enable-ssa
# should we compute SSA stuff (I don't know why this is experimental)
# Answer: It was when it first came out, but should be built into GCC by now

AC_MSG_CHECKING([whether to use SSA Single Static Assignment form optimizations])

AC_ARG_ENABLE(ssa,
              [AC_HELP_STRING([--enable-ssa],
                             [use experimental SSA optimizations])],
              ,
              enable_ssa=no)

AC_MSG_RESULT($enable_ssa)

if test "x$enable_ssa" = xyes; then
   AC_LATE_WARN([you have selected to use experimental ssa optimizations. Most likely you should not use this])
   CXXFLAGS="$CXXFLAGS -fssa -fssa-opt -fssa-dce"
   CFLAGS="$CFLAGS -fssa -fssa-opt -fssa-dce"
fi


### --enable-prefetchla
# Seems to have little change on the code speed.
AC_MSG_CHECKING([whether to use prefetch on loop arrays optimization])

AC_ARG_ENABLE(prefetchla,
              [AC_HELP_STRING([--enable-prefetchls],
                             [use prefetching on loop arrays optimizations; results can be better or worse so test it])],
              ,
              enable_prefetchla=no)

AC_MSG_RESULT($enable_prefetchla)

if test "x$enable_prefetchla" = xyes; then
   AC_LATE_WARN([you have selected to use prefetch loop array optimizations which can make things worse])
   CXXFLAGS="$CXXFLAGS -fprefetch-loop-arrays"
   CFLAGS="$CFLAGS -fprefetch-loop-arrays"
fi


### --enable-ftreevec
# Seems to have little change on the code speed.
AC_MSG_CHECKING([whether to use loop vectorization on tree optimization])

AC_ARG_ENABLE(ftreevec,
              [AC_HELP_STRING([--enable-ftreevec],
                             [use loop vectorization on trees])],
              ,
              enable_ftreevec=no)

AC_MSG_RESULT($enable_ftreevec)

if test "x$enable_ftreevec" = xyes; then
   AC_LATE_WARN([You have selected to use loop vectorization on trees optimizations])
   CXXFLAGS="$CXXFLAGS -ftree-vectorize"
   CFLAGS="$CFLAGS -ftree-vectorize"
fi


### --enable-fivopts
# Seems to have little change on the code speed.
AC_MSG_CHECKING([whether to use induction variable optimization])

AC_ARG_ENABLE(fivopts,
              [AC_HELP_STRING([--enable-fivopts],
                             [use induction variable optimizations])],
              ,
              enable_fivopts=no)

AC_MSG_RESULT($enable_fivopts)

if test "x$enable_fivopts" = xyes; then
   AC_LATE_WARN([You have selected to use induction variable optimizations])
   CXXFLAGS="$CXXFLAGS -fivopts"
   CFLAGS="$CFLAGS -fivopts"
fi


### --enable-funsafeloop
# Seems to have little change on the code speed.
AC_MSG_CHECKING([whether to use unsafe loop optimization])

AC_ARG_ENABLE(funsafeloop,
              [AC_HELP_STRING([--enable-funsafeloop],
                             [use unsafe loop optimization])],
              ,
              enable_funsafeloop=no)

AC_MSG_RESULT($enable_funsafeloop)

if test "x$enable_funsafeloop" = xyes; then
   AC_LATE_WARN([You have selected to use unsafe loop optimizations])
   CXXFLAGS="$CXXFLAGS -funsafe-loop-optimizations"
   CFLAGS="$CFLAGS -funsafe-loop-optimizations"
fi


### --enable-mem-debug

AC_MSG_CHECKING([whether to enable memory debug checks])

AC_ARG_ENABLE(mem-debug,
              [AC_HELP_STRING([--enable-mem-debug],
                              [enable array-bounds and other memory debug checks])],
              ,
              enable_mem_debug=no)

if test "x$enable_mem_debug" = xyes; then
   AC_DEFINE(INVT_MEM_DEBUG,1,[enable memory checking?])
   AC_DEFINE(GVX_MEM_DEBUG,1,[enable memory checking?])
fi

AC_MSG_RESULT($enable_mem_debug)



### --enable-debug

AC_MSG_CHECKING([whether to enable debug information with -g])

AC_ARG_ENABLE(debug,
              [AC_HELP_STRING([--disable-debug],
                              [disable debug information (i.e., turn off -g)])],
              ,
              enable_debug=yes)

# we don't immediately add -g to CXXFLAGS; instead we do that at the
# very end of the configure script

AC_MSG_RESULT($enable_debug)



### --enable-profiling

AC_MSG_CHECKING([whether to enable profiling with -pg])

AC_ARG_ENABLE(profiling,
              [AC_HELP_STRING([--enable-profiling],
                              [enable code profiling with gprof (i.e., turn on -pg)])],
              ,
              enable_profiling=no)

if test "x$enable_profiling" = xyes; then
   CXXFLAGS="$CXXFLAGS -pg"
   CFLAGS="$CFLAGS -pg"
fi

AC_MSG_RESULT($enable_profiling)



### --enable-tracing

AC_MSG_CHECKING([whether to enable GVX_TRACE statements])

AC_ARG_ENABLE(tracing,
              [AC_HELP_STRING([--disable-tracing],
                              [enable code profiling and tracing with GVX_TRACE statements])],
              ,
              enable_tracing=yes)

if test "x$enable_tracing" = xno; then
   AC_DEFINE(GVX_NO_PROF,1,[whether to disable GVX_TRACE statements])
fi

AC_MSG_RESULT($enable_tracing)



### --enable-warnings (--disable-warnings)

AC_MSG_CHECKING([whether to enable compiler warnings])

AC_ARG_ENABLE(warnings,
              [AC_HELP_STRING([--disable-warnings],
                              [turn off compiler warnings])],
              ,
              enable_warnings=yes)

if test "x$enable_warnings" = xyes; then
   CXXWARNINGS="$CXXWARNINGS -Wall"
   CWARNINGS="$CWARNINGS -Wall -Wstrict-prototypes -Wold-style-definition -Wmissing-prototypes -Wmissing-declarations -Wnested-externs -Wbad-function-cast"
else
   CXXWARNINGS="$CXXWARNINGS -w"
   CWARNINGS="$CWARNINGS -w"
fi

AC_MSG_RESULT($enable_warnings)




### --enable-optimization

AC_MSG_CHECKING([for desired optimization level])

AC_ARG_ENABLE(optimization,
              [AC_HELP_STRING([--enable-optimization],
                              [specify optimization level [O3]])],
              ,
              enable_optimization=-O3)

if test "x$enable_optimization" = xyes; then
   enable_optimization=-O3
fi

if test "x$enable_optimization" = xno; then
   enable_optimization=-O2
fi

CXXFLAGS="$CXXFLAGS $enable_optimization"
CFLAGS="$CFLAGS $enable_optimization"

AC_MSG_RESULT($enable_optimization)


### --enable-shlibs

# do dynamic (shared libs) or static linking?
AC_MSG_CHECKING([whether to use dynamic linking, with most objects built as shared libraries])

AC_ARG_ENABLE(shlibs,
              [AC_HELP_STRING([--enable-shlibs],
                              [disable dynamic linking and shared libs])],
              ,
              enable_shlibs=no)

# use $(o) as extension for shared object files (will be .o or .so);
# see towards the end of this file what happens when enable_shlibs is
# yes:
o="o"
AC_MSG_RESULT($enable_shlibs)

### --disable-quietcompile

# disable quiet compilation?
AC_MSG_CHECKING([whether to be quiet during compilation])

AC_ARG_ENABLE(quietcompile,
              [AC_HELP_STRING([--disable-quietcompile],
                              [disable quiet compilation])],
              ,
              enable_quietcompile=yes)
AC_MSG_RESULT($enable_quietcompile)

QUIETCOMPILE=$enable_quietcompile
AC_SUBST(QUIETCOMPILE)


### --disable-fink

AC_MSG_CHECKING([whether to use Fink packages, if available])

AC_ARG_ENABLE(fink,
              [AC_HELP_STRING([--disable-fink],
                              [disable automatic detection and use of packages
                               installed via Fink under Darwin (Mac OS X)
                               (default=use Fink if available)])],
              ,
              enable_fink=yes)

AC_MSG_RESULT($enable_fink)

### --enable-ldrelax
# Tell the linker to relax, this optimization is platform specific, but will
# either do nothing or optimize the code. However, debug symbols can get messed
# so do not use when debugging

AC_MSG_CHECKING([whether the Linker will use relax for global optimizations])

AC_ARG_ENABLE(ldrelax,
              [AC_HELP_STRING([--enable-ldrelax],
                             [use relax with the linker for optimizations, man ld for more details])],
              ,
              enable_ldrelax=no)

AC_MSG_RESULT($enable_ldrelax)

if test "x$enable_ldrelax" = xyes; then
   AC_LATE_WARN([you have selected to use linker relax. This might mangle debuging symbols. Dont use when debugging])
   CXXFLAGS="$CXXFLAGS -Xlinker --relax"
   CFLAGS="$CFLAGS -Xlinker --relax"
fi

### --enable-mfpmathboth
# Use both sse and 387 registers for floating point math. This is considered experimental.
# This will essentially double the amount of usable registers so in theory it should speed things up

AC_MSG_CHECKING([whether to use both SSE and 387 floating point registers])

AC_ARG_ENABLE(mfpmathboth,
              [AC_HELP_STRING([--enable-mfpmathboth],
			     [use both SSE and 387 registers for floating point math])],
	      ,
	      enable_mfpmathboth=no)

AC_MSG_RESULT($enable_mfpmathboth)

if test "x$enable_mfpmathboth" = xyes; then
   AC_LATE_WARN([you have selected to use both SSE and 387 registers for floating point math. This is experimental])
   CXXFLAGS="$CXXFLAGS -mfpmath=sse,387"
   CFLAGS="$CFLAGS -mfpmath=sse,387"
fi

### --enable-cuda

AC_MSG_CHECKING([whether to enable nVidia CUDA GPU optimizations])

AC_ARG_ENABLE(cuda,
              [AC_HELP_STRING([--enable-cuda],
                              [enable nVidia CUDA GPU optimizations])],
              ,
              enable_cuda=no)

if test "x$enable_cuda" = xyes; then
   AC_DEFINE(INVT_USE_CUDA,1,[enable nVidia CUDA GPU optimizations?])
fi
AC_MSG_RESULT($enable_cuda)

### --disable-cpp11

AC_MSG_CHECKING([whether to enable c++11 features])

AC_ARG_ENABLE(cpp11,
              [AC_HELP_STRING([--disable-cpp11],
                              [disable c++11 features])],
              ,
              enable_cpp11=yes)

if test "x$enable_cpp11" = xyes; then
   AC_DEFINE(INVT_USE_CPP11,1,[enable c++11 features])
fi

AC_MSG_RESULT($enable_cpp11)


##########################################################################
### Checks for programs

AC_LANG(C++)

# Compilers: g++-2.x cannot compile the code anymore, and you
# need to have a 3.x version of g++ installed. To allow for
# multiple revisions (e.g., 3.0.2 in Mandrake 8.1 and 3.0.4 in
# Mandrake 8.2), you should create generic symbolic links named
# g++-3 and gcc-3; for example, if you have g++-3.0.4 installed:
# ln -s /usr/bin/g++-3.0.4 /usr/bin/g++-3
# ln -s /usr/bin/gcc-3.0.4 /usr/bin/gcc-3

AC_PROG_CC(gcc gcc-3 gcc3 cc)
AC_PROG_CXX(g++ g++-3 g++3 c++ CC)


# if cell processor, check for ppu-gcc, ppu-g++, spu-gcc, spu-g++
if test "x$cpu_model" = "xcell"; then
   AC_PATH_PROGS([PPUCC], [ppu-gcc], [])
   if test "x$PPUCC" = "x"; then
      AC_LATE_WARN([You need 'ppu-gcc' in your path to be able to compile
        code for the Cell processor. It is part of the IBM Cell SDK.])
   fi
   AC_SUBST(PPUCC)

   AC_PATH_PROGS([PPUCXX], [ppu-g++], [])
   if test "x$PPUCXX" = "x"; then
      AC_LATE_WARN([You need 'ppu-g++' in your path to be able to compile
        code for the Cell processor. It is part of the IBM Cell SDK.])
   fi
   AC_SUBST(PPUCXX)

   AC_PATH_PROGS([SPUCC], [spu-gcc], [])
   if test "x$SPUCC" = "x"; then
      AC_LATE_WARN([You need 'spu-gcc' in your path to be able to compile
        code for the Cell processor. It is part of the IBM Cell SDK.])
   fi
   AC_SUBST(SPUCC)

   AC_PATH_PROGS([SPUCXX], [spu-g++], [])
   if test "x$SPUCXX" = "x"; then
      AC_LATE_WARN([You need 'spu-g++' in your path to be able to compile
        code for the Cell processor. It is part of the IBM Cell SDK.])
   fi
   AC_SUBST(SPUCXX)

   AC_PATH_PROGS([EMBEDSPU], [embedspu], [])
   if test "x$EMBEDSPU" = "x"; then
      AC_LATE_WARN([You need 'embedspu' in your path to be able to compile
        code for the Cell processor. It is part of the IBM Cell SDK.])
   fi
   AC_SUBST(EMBEDSPU)
fi

# if using CUDA, look for the nvcc compiler and CUDA libs:
if test "x$enable_cuda" = xyes; then
   AC_PATH_PROGS([NVCC], [nvcc], [], [${PATH}:/usr/local/cuda/bin])
   if test "x${NVCC}" = "x"; then
      AC_LATE_WARN([You need 'nvcc' in your path to be able to compile
        code for CUDA/GPU. It is part of the nVidia CUDA SDK.])
   else
      if test "x${NVCC}" = "x/usr/bin/nvcc"; then
        ### looks like an "official" cuda distro (RPMs), do proper checking for libs:
	NVCCFLAGS=""

	# well, if we have nvcc in /usr/bin we here just assume that we also have the cuda libs...
      	CUDA_LIBS="-lcufft -lcublas"
      	CUDA_RT_LIBS="-lcudart"

      else
        ### old hacky way of finding cuda libs that had been installed by hand

        # get directory of nvcc, then pop the last dir level (/.../bin)
      	cudadir=`dirname "${NVCC}"`
      	cudadir="${cudadir%/*}"
      	glew_inc_dir="/usr/local/cudasdk/C/common/inc/"
      	glew_lib_dir="/usr/local/cudasdk/C/common/lib/linux/"
      	cutil_lib_dir="/usr/local/cudasdk/C/lib/"

      	# make sure we find our includes
      	NVCCFLAGS="-I ${cudadir}/include"

      	# FIXME: should test for these...
      	LDFLAGS="$LDFLAGS -L${cudadir}/lib${bitsfx} -L${glew_lib_dir} -L${cutil_lib_dir}"
      	CUDA_LIBS="-lcufft -lcublas"
      	CUDA_RT_LIBS="-lcudart"
      	GLEW_LIBS="-lGLEW_x86_64"
      	CUTIL_LIBS="-lcutil"
      	CPPFLAGS="-I${cudadir}/include $CPPFLAGS -I${glew_inc_dir}"
      fi
   fi
fi
AC_SUBST(NVCC)
AC_SUBST(NVCCFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_RT_LIBS)
AC_SUBST(GLEW_LIBS)
AC_SUBST(CUTIL_LIBS)

# We'll set this to 'yes' if we find that we're using a gcc version
# that supports precompiled headers.
precompiled_headers=no

# what CPU optimizations are to be used ?
case `basename ${CXX}` in
   g++*)
      AC_MSG_CHECKING([for gcc version])
      gcc_version=`${CXX} -dumpversion`
      AC_MSG_RESULT($gcc_version)
      CFLAGS="$CFLAGS -std=gnu99"
      case $gcc_version in
         2.*)
            AC_LATE_WARN([some compilations will probably fail with gcc < 3.0])
            ;;
         3.3*)
            # FSF gcc 3.3 doesn't have precompiled headers, but
            # Apple's gcc 3.3 DOES have precompiled headers:
            if g++ -v 2>&1 | grep -q Apple; then
               precompiled_headers=yes
            fi
            ;;
         3.4.*)
            precompiled_headers=yes
            ;;
         4.0*)
            # avoid an overzealous -Wnon-virtual-dtor that g++ 4.0.x
            # gives when compiling Qt 3.x
            CXXFLAGS="$CXXFLAGS -Wno-non-virtual-dtor"
            precompiled_headers=yes
            ;;
         4.1*)
            # avoid an overzealous -Wnon-virtual-dtor that g++ 4.0.x
            # gives when compiling Qt 3.x
            CXXFLAGS="$CXXFLAGS -Wno-non-virtual-dtor"
            precompiled_headers=yes
            ;;
         4.2*)
            precompiled_headers=yes
            ;;
         4.3*)
            precompiled_headers=yes
	    if test "x$enable_cpp11" = xyes; then
	       CXXFLAGS="$CXXFLAGS -std=c++0x"
	    fi
	    CPPFLAGS="$CPPFLAGS -D__STDC_UTF_16__" # for matlab, gcc-4.7 defines it, though
            ;;
         4.4*)
            precompiled_headers=yes
	    if test "x$enable_cpp11" = xyes; then
	       CXXFLAGS="$CXXFLAGS -std=c++0x"
	    fi
	    CPPFLAGS="$CPPFLAGS -D__STDC_UTF_16__" # for matlab, gcc-4.7 defines it, though
            ;;
         4.5*)
            precompiled_headers=yes
	    if test "x$enable_cpp11" = xyes; then
	       CXXFLAGS="$CXXFLAGS -std=c++0x"
	    fi
	    CPPFLAGS="$CPPFLAGS -D__STDC_UTF_16__" # for matlab, gcc-4.7 defines it, though
            ;;
         4.6*)
            precompiled_headers=yes
	    if test "x$enable_cpp11" = xyes; then
	       CXXFLAGS="$CXXFLAGS -std=c++0x"
	    fi
	    CPPFLAGS="$CPPFLAGS -D__STDC_UTF_16__" # for matlab, gcc-4.7 defines it, though
            ;;
         4.7*)
            precompiled_headers=yes
	    if test "x$enable_cpp11" = xyes; then
	       CXXFLAGS="$CXXFLAGS -std=c++11"
	    fi
	    ###CXXFLAGS="$CXXFLAGS -fno-weak"
            ;;
         4.8*)
            precompiled_headers=yes
	    if test "x$enable_cpp11" = xyes; then
	       CXXFLAGS="$CXXFLAGS -std=c++11"
	    fi
            ;;
      esac
      archflag=""
      if test "x$enable_aggressive" = xyes; then
          if test "x$cpu_supports_mmx" = xyes; then
             CXXFLAGS="$CXXFLAGS -mmmx"
             CFLAGS="$CFLAGS -mmmx"
          fi
          if test "x$cpu_supports_sse" = xyes; then
             CXXFLAGS="$CXXFLAGS -msse"
             CFLAGS="$CFLAGS -msse"
          fi
          if test "x$cpu_supports_sse2" = xyes; then
             CXXFLAGS="$CXXFLAGS -msse2"
             CFLAGS="$CFLAGS -msse2"
          fi
          if test "x$cpu_supports_sse3" = xyes; then
             CXXFLAGS="$CXXFLAGS -msse3"
             CFLAGS="$CFLAGS -msse3"
          fi
         if test "x$cpu_supports_3dnow" = xyes; then
            CXXFLAGS="$CXXFLAGS -m3dnow"
            CFLAGS="$CFLAGS -m3dnow"
         fi

         case $cpu_model in
            pentium*) archflag="-march=$cpu_model" ;;
            athlon*) archflag="-march=$cpu_model" ;;
         esac
      else
         # not using aggressive optimizations
         case $uname_p in
            i686) archflag="-march=i686" ;;
         esac
      fi

      if test "x$cpu_model" = xopteron; then
        if test "x$enable_force32" = xyes; then
            # AMD recomended compiler 32 bit switches
            archflag="-march=i686 -m32"
        else
            # AMD recomended compiler 64 bit switches
            CXXFLAGS="$CXXFLAGS -m64 -march=k8 -funroll-all-loops -funit-at-a-time -fpeel-loops -ftracer -funswitch-loops"
            CFLAGS="$CFLAGS -m64 -march=k8 -funroll-all-loops -funit-at-a-time -fpeel-loops -ftracer -funswitch-loops"
        fi
      fi

      if test "x$cpu_model" = xopteron; then
        if test "x$enable_force32" = xyes; then
            # AMD recomended compiler 32 bit switches
            archflag="-march=i686 -m32"
        else
            # AMD recomended compiler 64 bit switches
            CXXFLAGS="$CXXFLAGS -m64 -march=k8 -funroll-all-loops -funit-at-a-time -fpeel-loops -ftracer -funswitch-loops"
            CFLAGS="$CFLAGS -m64 -march=k8 -funroll-all-loops -funit-at-a-time -fpeel-loops -ftracer -funswitch-loops"
        fi
      fi


      if test "x$cpu_model" = xcore2; then
      	 case $gcc_version in
         4.3*) archflag="-march=core2";;
	 *) if test "x$enable_force32" = xyes; then
	 archflag="-march=i686 -m32"
	 else
	 archflag="-march=nocona"
	 fi
	 ;;
	 esac
      fi
      	 

      if test "x$uname_p" = xpowerpc -o "x$uname_p" = xppc64 -o "x$uname_p" = xppc ; then
        AC_MSG_CHECKING([whether to use altivec])
        CXXFLAGS_SAVE=$CXXFLAGS
        CFLAGS_SAVE=$CFLAGS

        # On the cell processor, try -mabi=altivec -maltivec, otherwise -faltivec:
        if test "x$cpu_model" = xcell; then
           CXXFLAGS="$CXXFLAGS -maltivec -mabi=altivec -funroll-all-loops"
           CFLAGS="$CFLAGS -maltivec -mabi=altivec -funroll-all-loops"
        else
           # Apple gcc compiler
           CXXFLAGS="$CXXFLAGS -faltivec"
           CFLAGS="$CFLAGS -faltivec"
        fi

        AC_TRY_COMPILE([#include <altivec.h>],
                       [vector int foo; vec_ld(0, (int*) 0);],
                       [AC_MSG_RESULT([yes])
                        AC_DEFINE(INVT_USE_ALTIVEC,1,[enable PowerPC Altivec optimizations?])],
                       [AC_MSG_RESULT([no])
                        CXXFLAGS=$CXXFLAGS_SAVE
                        CFLAGS=$CFLAGS_SAVE])
      fi
      CXXFLAGS="$CXXFLAGS $archflag"
      CFLAGS="$CFLAGS $archflag"
      ;;
   *)
      AC_LATE_WARN([unknown compiler ${CXX}; some gcc-specific settings may fail])
      ;;
esac

# check for aclocal program:
AC_PATH_PROGS([ACLOCAL_PROG], [aclocal], [])
AC_ARG_VAR([ACLOCAL_PROG], [full path to aclocal program])

# check autoconf version
AC_PATH_PROGS([AUTOCONF_PROG], [autoconf-2.5x autoconf-2.53 autoconf], [])
if test "x$AUTOCONF_PROG" = "x"; then
   AUTOCONF_PROG=${buildscripts}/autoconf_warning.sh
else
   autoconf_version=`$AUTOCONF_PROG --version | head -1 | awk '{print $NF}'`
   AC_MSG_CHECKING([for autoconf version])
   case $autoconf_version in
      2.5[[3456789]]*|2.[[6789]]*) # need double set of brackets:
                      #    outer level is stripped by m4 parsing,
                      #    inner level remains to be seen as a glob pattern
         AC_MSG_RESULT([OK, $autoconf_version >= 2.53])
         ;;
      *)
         AC_MSG_RESULT([$autoconf_version < 2.53; autoconf will be disabled])
         AUTOCONF_PROG=${buildscripts}/autoconf_warning.sh
         ;;
   esac
fi
AC_ARG_VAR([AUTOCONF_PROG], [full path to autoconf program])

##############
# lex and yacc
AC_PROG_LEX
if test "x$LEX" = "xflex" ; then
   AC_SUBST([LFLAGS], [''])
else
   AC_LATE_WARN([flex seems to be missing; some targets/modules may not build])
fi

AC_PROG_YACC
if echo "x$YACC" | grep -q bison ; then
   AC_SUBST([YACC], [bison]) # skip -y option
else
   AC_LATE_WARN([bison seems to be missing; some targets/modules may not build])
fi

# check for Matlab libs
AC_MSG_CHECKING([whether to enable matlab mex files])
AC_ARG_ENABLE(matlab,
         [AC_HELP_STRING([--disable-matlab],
               [disable matlab mex files])],
         ,
         enable_matlab=yes)
AC_MSG_RESULT($enable_matlab)

# test for mex
if test "x$enable_matlab" = xyes; then
   AC_PATH_PROGS([MEX_PROG], [mex], [], [${PATH}:/lab/local/bin])
   AC_SUBST(MEX_PROG)

   if test "x$MEX_PROG" != "x"; then
      # MEX file extension (dependent on operating system)
      # Linux:
      MXEXT=mexglx
      AC_SUBST(MXEXT)

      # Use fPIC (Position Independant Code) so that shared libs work with mex
      # This may or may not create a performance hit. However, if you are not using matlab code
      # Then just disable it and the switch goes away.
      AC_LATE_WARN([Use of Matlab and Mex shared libs forces Position Independant Code via -fPIC option])
      CXXFLAGS="$CXXFLAGS -fPIC"
      CFLAGS="$CFLAGS -fPIC"
   fi
fi

# Enable users to specify where Matlab is installed. The default is to
# fall back on iLab's installation path.
AC_ARG_WITH([matlabroot],
            [AC_HELP_STRING([--with-matlabroot=DIR],
                            [where Matlab is installed (e.g., /lab/local/matlabR2007a) [default=no]])],
            [matlab_root="$with_matlabroot"],
            [matlab_root="no"])

with_matp=no

# try to locate matlab root, skip if supplied by user:
if test "x$matlab_root" = "xno" ; then
   AC_MSG_CHECKING([location of matlab root])
   for f in /lab/local/matlabR2011b \
            /lab/local/matlabR2010b \
	    /usr/local/matlab \
	    /opt/matlab \
	    /usr/local/matlabR2011b \
	    /usr/local/matlabR2010b ; do
     if test -d $f; then matlab_root=$f; break; fi
   done
   AC_MSG_RESULT($matlab_root)
fi

# Use the supplied (or default) Matlab root directory to appropriately
# set various build flags so that the compiler and linker can find the
# necessary headers and libraries.
MATLAB_LIBS=""
MATLAB_VPATH=""

if test "x$enable_matlab" = xyes; then
 if test "x$matlab_root" != "xno" ; then
   if test -d "$matlab_root"
   then
      matlab_incdir="$matlab_root/extern/include"
      if test "x$bitsfx" = "x64" ; then
         matlab_libdir="$matlab_root/bin/glnxa64"
      else
         matlab_libdir="$matlab_root/bin/glnx86"
      fi

      cppflags_save="$CPPFLAGS"
      ldflags_save="$LDFLAGS"
      CPPFLAGS="$CPPFLAGS -I$matlab_incdir" ### -D__STDC_UTF_16__ moved to compiler selection
      LDFLAGS="$LDFLAGS -L$matlab_libdir -Wl,-rpath-link,$matlab_libdir"

      # NOTE: if you have problems with ld not finding lhdf5, you need to fix your matlab install like this:
      # root@iLab:/lab/local/matlabR2010a/bin/glnxa64# ln -s libhdf5.so.6.0.2 libhdf5.so
      # root@iLab:/lab/local/matlabR2010a/bin/glnxa64# ln -s libhdf5_hl.so.6.0.2 libhdf5_hl.so

      AC_CHECK_LIB_CXX(mat, [#include <mat.h>], [matOpen("foo.mat", "w") ;],
                       [AC_SUBST([MATLAB_LIBS], ["-leng -lmx"])
                        AC_DEFINE([MATLAB_LIBS_AVAILABLE], 1, [if defined, Matlab functionality will not be NOP-ed out])
                        LDFLAGS="$ldflags_save"  # do not pollute LDFLAGS to avoid conflicts with Qt, fftw, etc. Will use VPATH
			AC_SUBST([MATLAB_VPATH], ["$matlab_libdir"])
                        with_matp=yes],
                       [AC_LATE_WARN([libmat missing])
                        CPPFLAGS="$cppflags_save"
                        LDFLAGS="$ldflags_save"],
			[-leng -lm -lut]);# -lmwfl -licudata -licuuc -licui18n -licuio -licuio -lhdf5 -lhdf5_hl -lmx]);
   else
      AC_LATE_WARN([Matlab root directory "$matlab_root" missing])
   fi
 fi
fi

AC_SUBST(with_matp)


# test for python, tclsh and env, needed by various scripts during make
AC_PATH_PROGS([PYTHON_PROG], [python], [])
if test "x$PYTHON_PROG" = "x"; then
      AC_LATE_WARN([You need 'python' in your path to be able to compile.
        It is one of the standard programming languages available for Linux.])
fi

AC_PATH_PROGS([TCLSH_PROG], [tclsh], [])
if test "x$TCLSH_PROG" = "x"; then
      AC_LATE_WARN([You need 'tclsh' in your path to be able to compile.
        It is the standard interpreter shell for Tcl language.])
fi

AC_PATH_PROGS([ENV_PROG], [env], [], /usr/bin)
if test "x$ENV_PROG" = "x"; then
      AC_LATE_WARN([You need '/usr/bin/env' to be able to compile.])
fi

AC_PATH_PROGS([MPEGENCODE_PROG], [mpeg_encode ppmtompeg], [], [])
if test "x$MPEGENCODE_PROG" = "x"; then
      AC_LATE_WARN([mpeg_encode program missing; high-quality mpeg-1 encoding will be unavailable])
else
      AC_DEFINE_UNQUOTED(MPEGENCODE_PROG, "$MPEGENCODE_PROG",
                         [path to mpeg_encode, if it exists])
fi

AC_PATH_PROGS([PAGER_PROG], [less more], [], [])
if test "x$PAGER_PROG" != "x"; then
      AC_DEFINE_UNQUOTED(PAGER_PROG, "$PAGER_PROG",
                         [path to pager program (less/more), if it exists])
fi

# check for TCL version >= 8.6, where they fixed const bugs in their API:
if test -x "/usr/bin/tclsh8.6" ; then
   AC_DEFINE(INVT_NEWER_TCL_VERSION,1,[use newer version of TCL (>=8.6)?])
fi

##########################################################################
### Checks for compiler characteristics


# If we think the compiler can do precompiled headers according to its
# version, then let's double-check that precompiled headers actually
# work and don't crash the compiler.
if test "x$precompiled_headers" = "xyes"; then
   AC_MSG_CHECKING([whether $CXX can generate usable precompiled headers])
   tmpheader="/tmp/autoconf-tmp-$$.h"
   echo '#include <algorithm>' >> $tmpheader
   echo '#include <deque>' >> $tmpheader
   echo '#include <iostream>' >> $tmpheader
   echo '#include <list>' >> $tmpheader
   echo '#include <map>' >> $tmpheader
   echo '#include <string>' >> $tmpheader
   echo '#include <vector>' >> $tmpheader
   $CXX $DEFS $CPPFLAGS $CXXFLAGS $CXXWARNINGS -x c++ -c $tmpheader -o ${tmpheader}.gch > /dev/null 2>&1
   result=$?
   if test $result -eq 0 && test -f ${tmpheader}.gch; then
      # OK, we got a .gch file; now let's make sure it works:
      echo "#include \"$tmpheader\"" \
         | $CXX $DEFS $CPPFLAGS $CXXFLAGS $CXXWARNINGS \
         -x c++ -c -o ${tmpheader}.o - > /dev/null 2>&1
      result=$?

      if test $result -eq 0; then
         AC_MSG_RESULT([yes])
      else
         AC_MSG_RESULT([no (.gch file was unusable); turning off precompiled headers])
         precompiled_headers=no
      fi
   else
      AC_MSG_RESULT([no (.gch file was not generated); turning off precompiled headers])
      precompiled_headers=no
   fi

   rm -f $tmpheader ${tmpheader}.gch ${tmpheader}.o

fi

# Check if the user wants to enable/disable precompiled headers
# (independent of whether the compiler actually supports precompiled
# headers):
AC_ARG_ENABLE(pch,
              [AC_HELP_STRING([--disable-pch],
                              [disable pre-compiled headers
                               (default=use pch if available)])],
              ,
              enable_pch=yes)

if test "x$enable_pch" != xyes; then
   precompiled_headers=no
fi

AC_MSG_CHECKING([whether to use precompiled headers])
AC_MSG_RESULT([$precompiled_headers])

AC_SUBST(precompiled_headers)


### --enable-werror (--disable-werror)

AC_MSG_CHECKING([whether to force compiler warnings to be errors])

AC_ARG_ENABLE(werror,
              [AC_HELP_STRING([--enable-werror],
                              [do not treat compiler warnings as errors])],
              ,
              enable_werror=no)

AC_MSG_RESULT($enable_werror)

if test "x$enable_werror" = xyes; then

   CWARNINGS="$CWARNINGS -Werror"

   AC_MSG_CHECKING([whether the c++ std library compiles without warnings])
   CXXFLAGS_SAVE=$CXXFLAGS
   CXXFLAGS="$CXXFLAGS $CXXWARNINGS -Werror"
   AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
#include <map>
#include <string>], [[
std::map<int, std::map<int, std::string> > m; m[0][0] = "";
]])],
                     [AC_MSG_RESULT([yes])
                      CXXWARNINGS="$CXXWARNINGS -Werror"],
                     [AC_MSG_RESULT([no, the compiler/library is buggy, so -Werror will be turned off])])
   CXXFLAGS=$CXXFLAGS_SAVE
fi


##########################################################################
### Checks for header files.

# Qt3 libraries (this check needs to come before the subsequent
# AC_CHECK_HEADERS() so that our Qt CPPFLAGS are set up before we try
# to check for <qwt/qwt_plot.h>, for example). :
gw_CHECK_QT3          # defined in macro file acsite.m4

# check for Qt4:
gw_CHECK_QT4          # defined in macro file acsite.m4


#### Check for fink on MacOSX
if test "x$enable_fink" = "xyes"; then
   if test -d /sw/include; then
      CPPFLAGS="$CPPFLAGS -I/sw/include"
      if test -d /sw/include/libpng; then
         CPPFLAGS="$CPPFLAGS -I/sw/include/libpng"
      fi
   fi

   if test -d /sw/lib; then
      LDFLAGS="$LDFLAGS -L/sw/lib"
   fi
fi

AC_DEFINE(__STDC_LIMIT_MACROS,1,[make C99 limit macros from <stdint.h> available in c++])
AC_DEFINE(__STDC_CONSTANT_MACROS,1,[make C99 constant macros from <stdint.h> available in c++])

# note, this list of files needs to have backslash-escaped newlines,
# because the list goes directly into a bash "for f in ...; do" loop
AC_CHECK_HEADERS([limits iosfwd iostream fstream sstream png.h \
                  popt.h values.h gsl/gsl_cblas.h gsl/gsl_linalg.h \
                  execinfo.h linux/videodev.h linux/videodev2.h \
                  fftw3.h \
                  ode/ode.h jpeglib.h net/ethernet.h arpa/inet.h \
                  QuickTime/QuickTime.h mpi.h GL/glut.h GL/osmesa.h \
                  linux/parport.h linux/ppdev.h linux/joystick.h \
                  fenv.h zlib.h bzlib.h qwt/qwt_plot.h lwpr/lwpr.h \
                  libkern/OSAtomic.h sys/io.h sys/soundcard.h SWI-Prolog.h \
                  libcwiimote/wiimote.h xclib/xcliball.h \
                  cxxabi.h libsvm-2.0/libsvm/svm.h GL/glew.h cuda_runtime.h usb.h sqlite3.h \
                  ],,,-)

# Newer versions of the FFMpeg suite install in their own seperate directories 
# (e.g. libavformat/avformat.h instead of ffmpeg/avformat.h. We need to check for
# that here to set an internal variable which we can use in later lib tests.
# Additionally, we'll define a generic INVT_HAVE_AV* and HAVE_[FFMPEG|LIB*]_* 
# variables for use in conditional compilation

#Check for avcodec.h in the ffmpeg directory
AC_CHECK_HEADER([ffmpeg/avcodec.h],
                [#Check for avcodec.h in the old style ffmpeg folder
                 AC_DEFINE(HAVE_FFMPEG_AVCODEC_H,,[avcodec.h was found in the ffmpeg directory]) 
                 AC_DEFINE(INVT_HAVE_AVCODEC,,[avcodec.h was found on this system]) 
                 AVCODEC_PATH=ffmpeg
                ],
                [#If we didn't find avcodec.h in the ffmpeg directory, then search for it in libavcodec
                 AC_CHECK_HEADER([libavcodec/avcodec.h],
                                 [
                                  AC_DEFINE(HAVE_LIBAVCODEC_AVCODEC_H,,[avcodec.h was found in the libavcodec directory]) 
                                  AC_DEFINE(INVT_HAVE_AVCODEC,,[avcodec.h was found on this system]) 
                                  AVCODEC_PATH=libavcodec
                                 ],
                                 [])
                 ])

AC_CHECK_HEADER([ffmpeg/avformat.h],
                [#Check for avformat.h in the old style ffmpeg folder
                 AC_DEFINE(HAVE_FFMPEG_AVFORMAT_H,,[avformat.h was found in the ffmpeg directory])
                 AC_DEFINE(INVT_HAVE_AVFORMAT,,[avformat.h was found on this system])
                 AVFORMAT_PATH=ffmpeg
                ],
                [#If we didn't find avformat.h in the ffmpeg directory, then search for it in libavformat
                 AC_CHECK_HEADER([libavformat/avformat.h],
                                 [
                                  AC_DEFINE(HAVE_LIBAVFORMAT_AVFORMAT_H,,[avformat.h was found in the libavformat directory])
                                  AC_DEFINE(INVT_HAVE_AVFORMAT,,[avformat.h was found on this system])
                                  AVFORMAT_PATH=libavformat
                                  ],
                                 [])
                 ])

AC_CHECK_HEADER([libswscale/swscale.h])

   
# Set up include paths. By convention, we use ADDINC to contain
# directories for searching for "user" includes within the build tree,
# and we use CPPFLAGS for directories for searching for <system>
# includes.

# NOTE: at the end of this script, we will add the contents of ADDINC
# to CPPFLAGS, so it is not necessary to do that again in the
# Makefile. We also export the separate variable ADDINC so that other
# programs that just want the include paths but no other cpp flags can
# use it. In practice, it should not be necessary for ADDINC to
# contain anything other than "-Isrc/"; if you think you need to add
# something else to ADDINC, check first if you can avoid it by just
# adding the directory name to the #include itself. This makes the
# source code clearer since we know that any #include "foo/bar/baz.h"
# always refers to ./src/foo/bar/baz.h. In the past, we had ADDINC
# also containing -Isrc/Qt and -Isrc/Qt/ui, for example, so it was not
# clear from looking at #include "foo.h" whether that refers to
# ./src/foo.h, ./src/Qt/foo.h, or ./src/Qt/ui/foo.h.

# Search the source directory -- see note above before adding anything
# else to ADDINC!
ADDINC="-I${source}"

##########################################################################
### Checks for functions and macros.

AC_MSG_CHECKING([whether mmap() is threadsafe])
case "$uname_s" in
   CYGWIN*)
      AC_MSG_RESULT([no])
      AC_DEFINE(MMAP_IS_NOT_THREADSAFE,1,[whether mmap() is threadsafe])
      ;;
   *)
      AC_MSG_RESULT([yes])
      ;;
esac

AC_DEFINE_IF_COMPILE([HAVE_FEENABLEEXCEPT],
[whether <fenv.h> has feenableexcept()],
[#include <fenv.h>], [feenableexcept(0);])

AC_DEFINE_IF_COMPILE([HAVE_STD_ISNAN],
[whether <cmath> has std::isnan()],
[#include <cmath>], [std::isnan(0.0);])

AC_DEFINE_IF_COMPILE([HAVE_STD_ISFINITE],
[whether <cmath> has std::isfinite()],
[#include <cmath>], [std::isfinite(0.0);])

AC_DEFINE_IF_COMPILE([HAVE_ISFINITE],
[whether <cmath> has isfinite()],
[#include <cmath>], [isfinite(0.0);])

AC_DEFINE_IF_COMPILE([HAVE___ISFINITEF],
[whether <cmath> has __isfinitef()],
[#include <cmath>], [__isfinitef(0.0);])

AC_DEFINE_IF_COMPILE([HAVE_SINCOS],
[whether <cmath> has sincos()],
[#include <cmath>], [sincos(0.0, 0, 0);])

AC_DEFINE_IF_RUN([HAVE_ASM_FSINCOS],
[whether we have the fsincos asm instruction],
[#include <cmath>],
[double arg=1.0;
 double sinarg, cosarg;
 asm("fsincos"
     :"=t"(cosarg), "=u"(sinarg)
     :"0"(arg));
 exit((cosarg == cos(arg) && sinarg == sin(arg)) ? 0 : 1);
])

AC_DEFINE_IF_COMPILE([HAVE_LGAMMA],
[whether <cmath> has lgamma()],
[#include <cmath>], [lgamma(0.0);])

AC_DEFINE_IF_COMPILE([HAVE_SRAND48],
[whether <cstdlib> has srand48()],
[#include <cstdlib>], [srand48(0);])

AC_DEFINE_IF_NOT_COMPILE([MISSING_SNPRINTF],
[whether <cstdio> is missing snprintf()],
[#include <cstdio>], [snprintf(0, 0, "");])

AC_DEFINE_IF_NOT_COMPILE([MISSING_M_PI],
[whether <cmath> is missing M_PI],
[#include <cmath>], [double v = M_PI;])

AC_DEFINE_IF_NOT_COMPILE([MISSING_M_PI_2],
[whether <cmath> is missing M_PI_2],
[#include <cmath>], [double v = M_PI_2;])

AC_DEFINE_IF_NOT_COMPILE([MISSING_M_E],
[whether <cmath> is missing M_E],
[#include <cmath>], [double v = M_E;])

AC_DEFINE_IF_NOT_COMPILE([MISSING_M_LN2],
[whether <cmath> is missing M_LN2],
[#include <cmath>], [double v = M_LN2;])

AC_DEFINE_IF_NOT_COMPILE([MISSING_CBRT],
[whether <cmath> is missing cbrt()],
[#include <cmath>], [double v = cbrt(1.0);])

AC_DEFINE_IF_NOT_COMPILE([MISSING_ISINF],
[whether <cmath> is missing isinf()],
[#include <cmath>], [int i = isinf(1.0);])

AC_DEFINE_IF_NOT_COMPILE([MISSING_ISNAN],
[whether <cmath> is missing isnan()],
[#include <cmath>], [int i = isnan(0.0);])

AC_DEFINE_IF_COMPILE([HAVE_PNG_ASM_FLAGS],
[whether <png.h> has png_get_asm_flags()],
[#include <png.h>], [
    png_uint_32 flags = png_get_asm_flags(0);
    png_uint_32 mask = png_get_asm_flagmask(PNG_SELECT_READ | PNG_SELECT_WRITE);
    png_set_asm_flags(0, flags | mask);
])

AC_DEFINE_IF_NOT_COMPILE([GVX_NO_READLINE],
[whether <readline/readline.h> is missing or broken],
[#include <stdio.h>
#include <readline/readline.h>], [void f(char*); rl_callback_handler_install("", f);])

AC_CHECK_MEMBERS([struct dirent.d_type],,,[
#include <sys/types.h>
#include <dirent.h>
])

AC_DEFINE_IF_COMPILE([HAVE_PTHREAD_SPINLOCK_T],
[whether pthread_spinlock_t is present and usable],
[#include <pthread.h>], [
    pthread_spinlock_t x;
    pthread_spin_init(&x, 0);
    pthread_spin_lock(&x);
    pthread_spin_unlock(&x);
])

AC_DEFINE_IF_COMPILE([HAVE_O_LARGEFILE],
[whether O_LARGEFILE is available for open() calls],
[
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
], [
    open("foo", O_WRONLY | O_LARGEFILE);
])

##########################################################################
### Checks for libraries.

###########################
# test for /usr/local/lib64 which some distros use and add if it exists.
# OpenCV and OpenSceneGraph need this
if test "x$bitsfx" = "x64"; then
   AC_MSG_CHECKING([for library directory /usr/local/lib64])
   if test -d /usr/local/lib64; then
      LDFLAGS="$LDFLAGS -L/usr/local/lib64"
      AC_MSG_RESULT([Added -L/usr/local/lib64 to LDFLAGS])
   else
      AC_MSG_RESULT([Not Found; not added to LDFLAGS])
   fi
fi

###########################
# test for /usr/local/lib which some distros use and add if it exists
AC_MSG_CHECKING([for library directory /usr/local/lib])
if test -d /usr/local/lib; then
   LDFLAGS="$LDFLAGS -L/usr/local/lib"
   AC_MSG_RESULT([Added -L/usr/local/lib to LDFLAGS])
else
   AC_MSG_RESULT([Not Found; not added to LDFLAGS])
fi

### --enable-ldsortcommon
# Tell the linker to sort common symbols by size
# This speeds up execution a little (Best baseline time 1.50.83 best time this option 1:47.51)

AC_MSG_CHECKING([whether the linker supports --sort-common])
LDFLAGS_SAVE=$LDFLAGS
LDFLAGS="$LDFLAGS -Wl,--sort-common"
AC_TRY_LINK([], [int i;],
            [linker_supports_sort_common=yes], [linker_supports_sort_common=no])
LDFLAGS=$LDFLAGS_SAVE
AC_MSG_RESULT($linker_supports_sort_common)

if test "x$linker_supports_sort_common" = xyes; then
   AC_MSG_CHECKING([whether the linker should use --sort-common])

   AC_ARG_ENABLE(ldsortcommon,
                 [AC_HELP_STRING([--enable-ldsortcommon],
                                 [The linker will sort common symbols by size])],
                 ,
                 enable_ldsortcommon=yes)

   AC_MSG_RESULT($enable_ldsortcommon)
else
   enable_ldsortcommon=no
fi

if test "x$enable_ldsortcommon" = xyes; then
   LDFLAGS="$LDFLAGS -Wl,--sort-common"
fi

### -rpath
AC_MSG_CHECKING([whether the linker supports -rpath])
LDFLAGS_SAVE=$LDFLAGS
LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/lib"
AC_TRY_LINK([], [int i;],
[linker_accepts_rpath=yes], [linker_accepts_rpath=no])
LDFLAGS=$LDFLAGS_SAVE
        AC_MSG_RESULT($linker_accepts_rpath)

###########################
# C math library
#if test $enable_ffastmath = no; then
AC_CHECK_LIB_CXX(m, [#include <cmath>], [sin(0.0)],
                  [LIBS="-lm $LIBS"], [AC_MSG_ERROR(libm missing)])
#else
#   LIBS="-lm $LIBS"
#fi

AC_CHECK_LIB(c, [ioperm],
             [AC_DEFINE(HAVE_IOPERM,1,[whether libc has ioperm()])])

###########################
# png image library
AC_CHECK_LIB_CXX_FULL(png, [INVT_HAVE_LIBPNG], [PNG_LIBS],
                      [png image reading/writing will be disabled],
                      [#include <png.h>], [int x = PNG_LIBPNG_VER])

###########################
# libjpeg image library
AC_CHECK_LIB_CXX_FULL(jpeg, [INVT_HAVE_LIBJPEG], [JPEG_LIBS],
                      [jpeg image reading will be disabled], [
#include <stdlib.h>
#include <stdio.h>
extern "C"
{
#include <jpeglib.h>
}
], [
jpeg_read_header(0, TRUE)
])

###########################
# X11 libraries
AC_PATH_XTRA

CPPFLAGS="$CPPFLAGS $X_CFLAGS"

# X_LIBS is now holding -L/usr/X11R6/lib or similar, but X_LIBS is a bit
# misnamed since -L options should really go in LDFLAGS variables. So, fix
# that up here:
LDFLAGS="$LDFLAGS $X_LIBS"
X_LIBS=""

AC_CHECK_LIB_CXX(X11, [#include <X11/Xlib.h>], [XOpenDisplay(0)],
                 [X_LIBS="-lX11 $X_LIBS"], [AC_LATE_WARN(libX11 missing)])

AC_CHECK_LIB_CXX(Xext,
                 [#include <X11/Xlib.h>
                  #include <X11/extensions/XShm.h>],
                 [XShmQueryExtension(0)],
                 [X_LIBS="-lXext $X_LIBS"], [AC_LATE_WARN(libXext missing)],
                 $X_LIBS)

AC_CHECK_LIB_CXX(Xtst,
                 [#include <X11/Xlib.h>
                  #include <X11/extensions/XTest.h>],
                 [XOpenDisplay(0)],
                 [X_LIBS="-lXtst $X_LIBS"], [AC_LATE_WARN(libXtst missing)],
                 $X_LIBS)

###########################
# tcl

AC_MSG_CHECKING([for tcl version])

if ! test -r ${with_tcl}/tclConfig.sh; then
   AC_LATE_WARN([could not find tclConfig.sh in ${with_tcl}])
   AC_MSG_RESULT([not found])
else
   TCL_VERSION=`sh -c ". ${with_tcl}/tclConfig.sh ; echo \\$TCL_VERSION"`
   TCL_SRC_DIR=`sh -c ". ${with_tcl}/tclConfig.sh ; echo \\$TCL_SRC_DIR"`
   TCL_LIB_VERSIONS_OK=`sh -c ". ${with_tcl}/tclConfig.sh ; echo \\$TCL_LIB_VERSIONS_OK"`
   TCL_INCLUDE_SPEC=`sh -c ". ${with_tcl}/tclConfig.sh ; echo \\$TCL_INCLUDE_SPEC"`
   CPPFLAGS="$TCL_INCLUDE_SPEC -I$TCL_SRC_DIR/generic $CPPFLAGS"
   AC_MSG_RESULT([$TCL_VERSION (from ${with_tcl}/tclConfig.sh)])
   AC_MSG_CHECKING([for tcl lib flag])
   TCL_LIBS="-ltcl${TCL_VERSION}"
   if test "x$TCL_LIB_VERSIONS_OK" != "xok"; then
      # remove dots from the version
      TCL_LIBS=`echo $TCL_LIBS | sed -e 's/\.//g'`
   fi
   AC_MSG_RESULT([$TCL_LIBS])
   AC_SUBST(TCL_LIBS)
fi

AC_CHECK_HEADERS([tclInt.h])

###########################
# tk

AC_MSG_CHECKING([for tk version])

if ! test -r ${with_tk}/tkConfig.sh; then
   AC_LATE_WARN([could not find tkConfig.sh in ${with_tk}])
   AC_MSG_RESULT([not found])
else
   TK_VERSION=`sh -c ". ${with_tk}/tkConfig.sh ; echo \\$TK_VERSION"`
   TK_SRC_DIR=`sh -c ". ${with_tk}/tkConfig.sh ; echo \\$TK_SRC_DIR"`
   TCL_LIB_VERSIONS_OK=`sh -c ". ${with_tcl}/tclConfig.sh ; echo \\$TCL_LIB_VERSIONS_OK"`
   CPPFLAGS="-I$TK_SRC_DIR/generic $CPPFLAGS"
   AC_MSG_RESULT([$TK_VERSION (from ${with_tk}/tkConfig.sh)])
   AC_MSG_CHECKING([for tk lib flag])
   TK_LIBS="-ltk${TK_VERSION}"
   if test "x$TCL_LIB_VERSIONS_OK" != "xok"; then
      # remove dots from the version
      TK_LIBS=`echo $TK_LIBS | sed -e 's/\.//g'`
   fi
   AC_MSG_RESULT([$TK_LIBS])
   AC_SUBST(TK_LIBS)
fi

###########################
# qwt plotting library
AC_CHECK_LIB_CXX_FULL(qwt, [INVT_HAVE_QWT], [QWT_LIBS],
                      [qwt plotting will be disabled],
                      [#include <cstddef>
                       #include <qwt/qwt_plot.h>],
                      [QwtPlot p; p.setMargin(10);],,,[$QT3_LIBS])

###########################
# popt library for command-line option processing
AC_CHECK_LIB_CXX_FULL(popt, [INVT_HAVE_LIBPOPT], [POPT_LIBS],
                      [no worries, libpopt isn't used currently],
                      [#include <popt.h>], [poptFreeContext(0)])

###########################
# bzip2 compression library
AC_CHECK_LIB_CXX_FULL(bz2, [INVT_HAVE_LIBBZ2], [BZIP2_LIBS],
                      [transparent bzip2 compression/decompression will be disabled],
                      [#include <bzlib.h>], [BZ2_bzReadOpen(0, 0, 0, 0, 0, 0)])

###########################
# Corba libs
AC_CHECK_LIB_CXX_ENABLED(omniORB4, [INVT_HAVE_CORBA], [CORBA_LIBS],
  [associated functionality will be disabled], [#include <omniORB4/CORBA.h>],
  [int c=0; char *v[] = {""}; CORBA::ORB_ptr orb = CORBA::ORB_init(c,v,"omniORB4");
   if (!CORBA::is_nil(orb)) orb->destroy();], , , [[-lomnithread]])

AC_PATH_PROGS([OMNIIDL], [omniidl], [])

###########################
# Ice libs
AC_CHECK_LIB_CXX_ENABLED(Ice, [INVT_HAVE_ICE], [ICE_LIBS],
  [associated functionality will be disabled], 
  [#include <cstddef>
   #include <Ice/Ice.h>],
  [int c=0; char *v[] = {""}; Ice::CommunicatorPtr ic =
   Ice::initialize(c,v); if (ic) ic->destroy();], , , [[-lIceUtil -lbz2]])

AC_PATH_PROGS([SLICE2CPP], [slice2cpp], [])
AC_PATH_PROGS([SLICE2JAVA], [slice2java], [])

###########################
# Boost libs

# --with-boost
AC_MSG_CHECKING([whether to use the Boost libraries])
AC_ARG_WITH(boost, [AC_HELP_STRING([--with-boost], [to use the Boost C++ libraries])], , with_boost="no")
if test "x$with_boost" = "xno"; then AC_MSG_RESULT([no]); with_boost=""; else AC_MSG_RESULT([$with_boost]); fi

if test "x$with_boost" = "xyes"; then
   ###########################
   # boost base library
   # note: AX_BOOST_BASE is defined is /usr/share/aclocal/ax_boost_base.m4
   # which is provided by the autoconf-archive RPM package, and finally it ends up in
   # our aclocal.m4 when we run the "aclocal" program.
   #
   # AX_BOOST_BASE calls: AC_SUBST(BOOST_CPPFLAGS) / AC_SUBST(BOOST_LDFLAGS)
   # and sets: HAVE_BOOST
   AC_CHECK_ACLOCAL_MACRO([AX_BOOST_BASE], [boost-base], [need to urpmi boost-devel])

   # boost serialization:
   # AX_BOOST_SERIALIZATION calls: AC_SUBST(BOOST_SERIALIZATION_LIB)
   # and sets: HAVE_BOOST_SERIALIZATION

   # note: for now the script seems to not correctly detect the lib, just use this:
   # ./configure --with-boost --with-boost-serialization=boost_serialization-mt

   AC_CHECK_ACLOCAL_MACRO([AX_BOOST_SERIALIZATION], [boost-serialization], [need to urpmi boost-devel])
fi

# Check if the Boost filesystem library is available
AC_CHECK_LIB_CXX_FULL(boost_filesystem,
                      [INVT_HAVE_BOOST_FILESYSTEM],
                      [BOOST_FILESYSTEM_LIB],
                      [libboost_filesystem dependent features will be disabled],
                      [#include <boost/filesystem/path.hpp>],
                      [boost::filesystem::path p("/foo/bar")],,,-lboost_system)

# Check if the Boost program_options library is available
AC_CHECK_LIB_CXX_FULL(boost_program_options,
                      [INVT_HAVE_BOOST_PROGRAM_OPTIONS],
                      [BOOST_PROGRAM_OPTIONS_LIB],
                      [libboost_program_options dependent features will be disabled],
                      [#include <boost/program_options.hpp>],
                      [boost::program_options::positional_options_description p],,,)

# Check if the Boost regex library is available
AC_CHECK_LIB_CXX_FULL(boost_regex,
                      [INVT_HAVE_BOOST_REGEX],
                      [BOOST_REGEX_LIB],
                      [libboost_regex dependent features will be disabled],
                      [#include <boost/regex.hpp>],
                      [boost::regex r("foo")],,,)

###########################
# libavcodec flags for programs using MPEGStream
AVCODEC_LIBS=""
LIBS_SAVE=$LIBS

# more recent versions of ffmpeg have libavutil, in addition to
# libavcodec and libavformat
AC_CHECK_LIB(avutil, av_add_q, [AVCODEC_LIBS="-lavutil -lpng"])

LIBS="$LIBS_SAVE $AVCODEC_LIBS"
AC_CHECK_LIB(avcodec, avcodec_get_edge_width,
        [AVCODEC_LIBS="-lavcodec $AVCODEC_LIBS"],
        [AC_LATE_WARN(libavcodec missing)])

LIBS="$LIBS_SAVE $AVCODEC_LIBS"
AC_CHECK_LIB(avformat, av_open_input_file,
        [AVCODEC_LIBS="-lavformat $AVCODEC_LIBS"],
        [AC_LATE_WARN(libavformat missing)])

LIBS="$LIBS_SAVE $AVCODEC_LIBS"
AC_CHECK_LIB(swscale, sws_getIdentityVec,
        [AVCODEC_LIBS="-lswscale $AVCODEC_LIBS"],
        [AC_LATE_WARN(libswscale missing)])

LIBS=$LIBS_SAVE
AC_SUBST(AVCODEC_LIBS)

###########################
# QuickTime libs: really simple hack for now -- just test if we're
# running on Darwin; if so, then add QuickTime and Carbon frameworks
# to the link line. Since those have to specified as "-framework" link
# options rather than "-l" options, we can't use the 'make' feature
# that allows '-lfoo' dependencies to be transformed to
# /full/path/to/libfoo.so. So, for now we just need to add the
# QuickTime and Carbon frameworks unconditionally to LIBS.
AC_MSG_CHECKING([for QuickTime libraries])
if test "x$uname_s" = xDarwin; then
   quicktime_ok=no

   # first we try linking against the QuickTime framework as is (that
   # will pull in the one from
   # /System/Library/Frameworks/QuickTime.framework/); if that fails
   # (because QuickTime>=7.0.4 with MacOSX 10.3.9 is broken if you try
   # to use libstdc++) then we use the 10.3.0 QuickTime framework from
   # /Developer/SDKs/MacOSX10.3.0.sdk/System/Library/Frameworks/QuickTime.framework/;
   # http://developer.mozilla.org/en/docs/Mac_OS_X_Build_Prerequisites
   # has more information on why we use an SDK to fix the problem

   LIBS_SAVE=$LIBS
   LIBS="-framework QuartzCore -framework QuickTime -framework Carbon $LIBS"
   AC_TRY_LINK([
#include <QuickTime/QuickTime.h>
#include <iostream>], [EnterMovies(); std::cout << 0;],
         [quicktime_ok=yes], [quicktime_ok=no])
   LIBS=$LIBS_SAVE

   if test "x$quicktime_ok" = xyes; then
      LIBS="-framework QuartzCore -framework QuickTime -framework Carbon $LIBS"
      AC_MSG_RESULT([-framework QuartzCore -framework QuickTime -framework Carbon])
   else
      LIBS_SAVE=$LIBS
      LIBS="-framework QuartzCore -framework QuickTime -framework Carbon $LIBS"
      LDFLAGS_SAVE=$LDFLAGS
      LDFLAGS="-F/Developer/SDKs/MacOSX10.3.0.sdk/System/Library/Frameworks $LDFLAGS"
      AC_TRY_LINK([
#include <QuickTime/QuickTime.h>
#include <iostream>], [EnterMovies(); std::cout << 0;],
         [quicktime_ok=yes], [quicktime_ok=no])
      LIBS=$LIBS_SAVE
      LDFLAGS=$LDFLAGS_SAVE

      if test "x$quicktime_ok" = xyes; then
         LIBS="-framework QuartzCore -framework QuickTime -framework Carbon $LIBS"
         LDFLAGS="-F/Developer/SDKs/MacOSX10.3.0.sdk/System/Library/Frameworks $LDFLAGS"
         AC_MSG_RESULT([-F/Developer/SDKs/MacOSX10.3.0.sdk/System/Library/Frameworks -framework QuartzCore -framework QuickTime -framework Carbon])
      else
         AC_MSG_RESULT([none])
      fi
   fi
else
   AC_MSG_RESULT([none])
fi

###########################
# LiveMedia

CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS -I/usr/include/liveMedia -I/usr/include/groupsock -I/usr/include/UsageEnvironment"

AC_CHECK_LIB_CXX_ENABLED(liveMedia_pic, [INVT_HAVE_LIVEMEDIA], [LIVEMEDIA_LIBS],
  [associated functionality will be disabled], [#include <liveMedia/liveMedia.hh>],
  [int c=0; char *v[] = {""}; Medium* client; 
  ], , CPPFLAGS=$CPPFLAGS_SAVE, [[-lgroupsock_pic -lBasicUsageEnvironment_pic -lUsageEnvironment_pic]])

###########################
# test a few API version differences in ffmpeg

AC_DEFINE_IF_COMPILE([INVT_FFMPEG_AVSTREAM_CODEC_IS_POINTER],
[whether AVStream::codec is a pointer],
[#include <$AVFORMAT_PATH/avformat.h>],
[extern AVStream* avs;
 AVCodecContext* cc = avs->codec;])

AC_DEFINE_IF_COMPILE([INVT_FFMPEG_AVCODECCONTEXT_HAS_TIME_BASE],
[whether <$AVCODEC_PATH/avcodec.h> has AVCodecContext::time_base],
[#include <$AVCODEC_PATH/avcodec.h>],
[extern AVCodecContext* cc;
 AVRational r = cc->time_base;])

AC_DEFINE_IF_COMPILE([INVT_FFMPEG_HAS_DEFAULTS_FUNCTIONS],
[whether <$AVCODEC_PATH/avcodec.h> has avcodec_get_frame_defaults() and avcodec_get_context_defaults()],
[#include <$AVCODEC_PATH/avcodec.h>],
[extern AVFrame* f; avcodec_get_frame_defaults(f);
 extern AVCodecContext* c; avcodec_get_context_defaults(c);])

AC_DEFINE_IF_COMPILE([INVT_FFMPEG_HAS_FORMATCONTEXT_FUNCTIONS],
[whether <$AVCODEC_PATH/avcodec.h> has av_alloc_format_context],
[#include <$AVCODEC_PATH/avcodec.h> 
 #include <$AVFORMAT_PATH/avformat.h>],
[AVFormatContext* fc = av_alloc_format_context();])

AC_DEFINE_IF_COMPILE([INVT_FFMPEG_URL_OPEN_FUNC_TAKES_SINGLE_POINTER],
[whether the url_ funcs take pointers],
[#include <$AVFORMAT_PATH/avformat.h>],
[extern ByteIOContext bi;
 url_fopen(&bi, "test.flv", URL_WRONLY);])

AC_DEFINE_IF_COMPILE([INVT_FFMPEG_AVFORMATCONTEXT_BYTEIO_ISPOINTER],
[whether the the ByteIO in AVFormatContext is a pointer],
[#include <$AVFORMAT_PATH/avformat.h>],
[AVFormatContext fc;
 ByteIOContext* cc = fc.pb;])


###########################
# fftw fast Fourier transform libs
AC_CHECK_LIB_CXX_FULL(fftw3, [INVT_HAVE_LIBFFTW3], [FFTW_LIBS],
                      [fft/ifft support will be disabled],
                      [#include <fftw3.h>], [int x=FFTW_FORWARD])

###########################
# libpqxx interface to PostgreSQL databases
AC_CHECK_LIB_CXX_FULL(pqxx, [INVT_HAVE_LIBPQXX], [PQXX_LIBS],
                      [PostgreSQL support via libpqxx will be disabled],
                      [#include <pqxx/pqxx>], [pqxx::largeobject x])

###########################
# libsvm support vector machine libs
AC_CHECK_LIB_CXX_FULL(svm, [INVT_HAVE_LIBSVM], [SVM_LIBS],
                      [svm support will be disabled],
                      [#include <libsvm-2.0/libsvm/svm.h>], [struct svm_problem sp;])

###########################
# gslcblas (GNU Scientific Library, C interface to Basic Linear Algebra Subprograms)
AC_CHECK_LIB_CXX_FULL(gslcblas, [INVT_HAVE_LIBGSLCBLAS], [GSLCBLAS_LIBS],
                      [gsl-based optimizations will be disabled],
                      [#include <gsl/gsl_cblas.h>],
                      [cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                         0, 0, 0, 0.0, 0, 0, 0, 0, 0.0, 0, 0)])
###########################
# gsl (GNU Scientific Library)
AC_CHECK_LIB_CXX_FULL(gsl, [INVT_HAVE_LIBGSL], [GSL_LIBS],
                      [gsl-based optimizations will be disabled],
                      [#include <gsl/gsl_matrix_double.h>],
                      [gsl_matrix_fprintf(0, 0, ".2f")],,,[$GSLCBLAS_LIBS])

###########################
# blitz++ (Object-Oriented Scientific Computing)
AC_CHECK_LIB_CXX_FULL(blitz, [INVT_HAVE_LIBBLITZ], [BLITZ_LIBS],
                      [Object-Oriented Scientific Computing will be disabled],
                      [#include <blitz/array.h>],
                      [blitz::Array<int,2> var])

# lapack (fortran-77 linear-algebra package; we try first without
# -lg2c, and if that fails we try again with -lg2c, and if that fails
# we try again with -lgfortran). NOTE that we have to do some tricky
# stuff to keep autoconf from outsmarting us -- we are doing several
# AC_CHECK_LIB()'s for the same library (lapack) -- if we check for
# the same symbol (e.g., dgesdd_) every time, then autoconf will think
# it's being smart and just use the cache for subsequent calls, even
# though that's not what we want because we'll have changed LDFLAGS in
# the meantime. So, to avoid that problem, we have to check for
# DIFFERENT symbols in each AC_CHECK_LIB() call.

LAPACK_LIBS=""
LDFLAGS_SAVE="$LDFLAGS"

lapack_ok=no
AC_CHECK_LIB(lapack, dgesdd_,
             [LAPACK_LIBS="-llapack -lblas"; lapack_ok=yes],,[-lblas])

if test $lapack_ok = no; then
   AC_CHECK_LIB(lapack, dgesvd_,
                   [LAPACK_LIBS="-llapack -lblas -lg2c"; lapack_ok=yes],,[-lblas -lg2c])
fi

if test $lapack_ok = no; then
   AC_CHECK_LIB(lapack, dggsvd_,
                   [LAPACK_LIBS="-llapack -lblas -lgfortran"; lapack_ok=yes],,[-lblas -lgfortran])
fi

if test $lapack_ok = yes; then
  AC_DEFINE(HAVE_LAPACK,1,[have fortran-77 linear-algebra library?])
else
  AC_LATE_WARN(liblapack missing; lapack-based optimizations will be disabled)
fi

AC_SUBST(LAPACK_LIBS)
LDFLAGS="$LDFLAGS_SAVE"
###########################
### --enable-fastmath-func
# check to use FastMathFunctions.H

AC_MSG_CHECKING([whether to use FastMathFunctions.H])

AC_ARG_ENABLE(fastmath_func,
              [AC_HELP_STRING([--enable-fastmath-func],
                             [use FastMathFunctions.H optimization])],
              ,
              enable_fastmath_func=no)

AC_MSG_RESULT($enable_fastmath_func)

if test "x$enable_fastmath_func" = xyes; then
   AC_LATE_WARN([you have selected FastMathFunctions.H which are less precise than the standard libm math functions])
   AC_DEFINE(INVT_USE_FASTMATH,1,[Use FastMathFunctions.H?])
   #AC_SUBST(INVT_USE_FASTMATH)
fi
###########################
# IEEE-1394 (FireWire) flags (for programs using the IEEE1394grabber):
AC_CHECK_LIB_CXX_FULL(raw1394, [INVT_HAVE_RAW1394], [RAW1394_LIBS],
                      [firewire support via libdc1394 1.x will be disabled],
                      [#include <libraw1394/raw1394.h>],
                      [raw1394_destroy_handle(0)])

# Make sure we have libdc1394, and the right version. The following
# Mandrake RPMs are known to work, and older Mandrake RPMs in which
# dc1394_dma_setup_capture takes fewer arguments are known NOT to
# work:
#
#  libdc1394_0-0.9.0-1mdk.i586.rpm
#  libdc1394_0-devel-0.9.0-1mdk.i586.rpm
#
# Now, newer versions also exist which take one more argument in
# dc1394_dma_setup_capture, so we check for both possibilities and add
# a flag IEEE1394_NEW to CFLAGS if we find the new version (this is
# necessary because the IEEE1394 include files do not contain any
# version number):

DC1394V1_LIBS=""
IEEE1394_NEW=0
DC1394WARN=0
AC_CHECK_LIB_CXX(dc1394_control, [#include <libdc1394/dc1394_control.h>],
                 [dc1394_dma_setup_capture(0,0,0,0,0,0,0,0,0,0,0)],
                 [DC1394V1_LIBS="-ldc1394_control"],
                 [DC1394WARN=`expr $DC1394WARN + 1`],
                 [$RAW1394_LIBS])
AC_CHECK_LIB_CXX(dc1394_control, [#include <libdc1394/dc1394_control.h>],
                 [dc1394_dma_setup_capture(0,0,0,0,0,0,0,0,0,0,0,0)],
                 [DC1394V1_LIBS="-ldc1394_control";
                 CPPFLAGS="$CPPFLAGS -DIEEE1394NEW"],
                 [DC1394WARN=`expr $DC1394WARN + 1`],
                 [$RAW1394_LIBS])

if test $DC1394WARN -lt 2; then
   AC_DEFINE(HAVE_IEEE1394,1,[have ieee1394 (firewire) library?])
   AC_FEATURE_STATUS([libdc1394_control], [ok],
    [firewire support ok via libdc1394 1.x (exclusive of 2.x)])
else
   AC_FEATURE_STATUS([libdc1394_control], [missing],
    [firewire support via libdc1934 1.x (exclusive of 2.x) will be disabled])
fi

AC_SUBST(DC1394V1_LIBS)

###########################
# libdc1394 2.x
AC_CHECK_LIB_CXX_FULL(dc1394, [HAVE_DC1394V2], [DC1394V2_LIBS],
                      [firewire support via libdc1934 2.x will be disabled],
                      [#include <dc1394/camera.h>],
                      [dc1394_camera_free((dc1394camera_t*)0)])

###########################
# libxml libraries:
AC_CHECK_PROG([have_libxml], [xml2-config], yes, no)

LIBXML_LIBS=""

if test "x$have_libxml" = xyes; then
   for opt in `xml2-config --cflags`; do
      case $opt in
         -I*) CPPFLAGS="$CPPFLAGS $opt" ;;
      esac
   done

   for opt in `xml2-config --libs`; do
      case $opt in
         -L*) LDFLAGS="$LDFLAGS $opt" ;;
         -l*) LIBXML_LIBS="$LIBXML_LIBS $opt" ;;
      esac
   done

   AC_DEFINE(HAVE_LIBXML,1,[have xml library?])
fi

AC_SUBST(LIBXML_LIBS)

###########################
# SDL libraries:
AC_PATH_PROGS([SDLCONFIG], [sdl-config], [])

if test "x$SDLCONFIG" != "x"; then
   AC_MSG_CHECKING([for $SDLCONFIG --libs])
   flags=`$SDLCONFIG --libs`
   AC_MSG_RESULT([$flags])
   next_arg_is_framework=no
   SDL_LIBS=""
   SDL_FRAMEWORKS=""
   for f in $flags; do
      if test $next_arg_is_framework = yes; then
         SDL_FRAMEWORKS="$SDL_FRAMEWORKS -framework $f"
         next_arg_is_framework=no
      else
         case "$f" in
            -l*)
               SDL_LIBS="$SDL_LIBS $f"
               next_arg_is_framework=no
               ;;
            -L*)
               LDFLAGS="$LDFLAGS $f"
               next_arg_is_framework=no
               ;;
            -framework)
               next_arg_is_framework=yes
               ;;
            -Wl,-framework*)
               SDL_FRAMEWORKS="$SDL_FRAMEWORKS $f"
               ;;
            *)
               AC_MSG_WARN([unknown flag in sdl-config --libs: $f])
               ;;
         esac
      fi
   done

   # On Mac OS X, we don't have any clean way to get the frameworks
   # only linked into the SDL apps, so for now we just stick the
   # frameworks in $LIBS where they will be linked in to all apps.
   LIBS="$SDL_FRAMEWORKS $LIBS"
else
   AC_LATE_WARN([libSDL missing])
fi

AC_SUBST(SDL_LIBS)

AC_CHECK_HEADERS([SDL/SDL.h])
AC_SUBST(with_sdl, [$ac_cv_header_SDL_SDL_h])

###########################
# SDL_mixer library:
AC_CHECK_LIB_CXX_FULL(SDL_mixer, [INVT_HAVE_LIBSDL_MIXER], [SDLMIX_LIBS],
                      [associated functionality will be disabled],
                      [#include <SDL/SDL_mixer.h>], [(void) Mix_GetMusicHookData()])

###########################
# SDL_gfx library:
AC_CHECK_LIB_CXX_FULL(SDL_gfx, [INVT_HAVE_LIBSDL_GFX], [SDLGFX_LIBS],
                      [associated functionality will be disabled],
                      [#include <SDL/SDL_gfxPrimitives.h>], [gfxPrimitivesSetFont(0, 0, 0)])

###########################
# SDL_image library:
AC_CHECK_LIB_CXX_FULL(SDL_image, [INVT_HAVE_LIBSDL_IMAGE], [SDLIMAGE_LIBS],
                      [associated functionality will be disabled],
                      [#include <SDL/SDL_image.h>], [(void) IMG_Load("foo")])

###########################
# OPENGL/Glut libraries:
if test "x$uname_s" = xDarwin; then

   # Mac OSX 10.5 has a weird bug where an unadorned link against -lGL
   # will give "ld: cycle in dylib re-exports with
   # /usr/X11/lib/libGL.dylib", so a verbose workaround is required
   # (see this thread:
   # http://lists.apple.com/archives/X11-users/2007/Oct/msg00130.html)
   LDFLAGS="$LDFLAGS -Wl,-dylib_file,/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib:/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"
fi

# OpenGL libGL
AC_CHECK_LIB_CXX_FULL(GL, [INVT_HAVE_LIBGL], [GL_LIBS],
                      [associated functionality will be disabled],
                      [#include <GL/gl.h>], [glBegin(GL_TRIANGLES)],,,[$X_LIBS])

# OpenGL libGLU
AC_CHECK_LIB_CXX_FULL(GLU, [INVT_HAVE_LIBGLU], [GLU_LIBS],
                      [associated functionality will be disabled],
                      [#include <GL/glu.h>], [gluOrtho2D(0,0,0,0)],,,[$GL_LIBS])

# OpenGL libGLUT
AC_CHECK_LIB_CXX_FULL(glut, [INVT_HAVE_LIBGLUT], [GLUT_LIBS],
                      [associated functionality will be disabled],
                      [#include <GL/glut.h>], [glutSetWindow(0)],,,[$GLU_LIBS])

# libGLEW:
AC_CHECK_LIB_CXX_FULL(GLEW, [INVT_HAVE_LIBGLEW], [GLEW_LIBS],
		[OpenGL Extension Wrangler (GLEW) will be disabled],
		[#include <GL/glew.h>], [glewInit()])

###########################
# OpenIL/libdevil: This library provides an easy-to-use, high-level
# interface for loading and saving images in a variety of file formats.
AC_CHECK_LIB_CXX_FULL(IL, [INVT_HAVE_LIBDEVIL], [DEVIL_LIBS],
                      [libdevil-dependent features will be disabled],
                      [#include <IL/il.h>],
                      [ilBindImage(1)],,,)


###########################
# Threading libraries:
AC_CHECK_LIB_CXX_FULL(pthread, [INVT_HAVE_LIBPTHREAD], [THREAD_LIBS],
                      [many programs will not work],
                      [#include <pthread.h>], [pthread_exit(0)])
LDFLAGS="$LDFLAGS -pthread"

###########################
# MPI libraries:
MPI_LIBS=""
AC_PATH_PROGS([MPICC_PROG], [mpic++], [])
if test "x$MPICC_PROG" = "x"; then
   AC_PATH_PROGS([MPICC_PROG], [mpiCC], [])
fi

if test "x$MPICC_PROG" = "x"; then
   AC_LATE_WARN([mpic++/mpiCC program missing; MPI programs won't be able to be built])
else
   libs=`$MPICC_PROG -showme:link 2> /dev/null`
   status=$?
   if test $status -ne 0; then
      libs=`$MPICC_PROG -show 2> /dev/null`
      status=$?
   fi
   if test $status -eq 0; then
      for arg in $libs; do
          case $arg in
             -l*)
                MPI_LIBS="$MPI_LIBS $arg"
                ;;
             -L*)
                LDFLAGS="$LDFLAGS $arg"
                ;;
          esac
      done
   fi
   AC_MSG_CHECKING([for MPI libraries])
   AC_MSG_RESULT([$MPI_LIBS])
   AC_DEFINE(INVT_HAVE_MPI,1,[have MPI library?])
fi
AC_SUBST(MPI_LIBS)

###########################
# gd library (for manipulating image files):
#AC_CHECK_LIB_CXX_FULL(gd, [INVT_HAVE_LIBGD], [GD_LIBS],
#                      [no worries, this is not currently used],
#                      [#include <gd.h>], [gdImageDestroy(0)])

###########################
# EyeLink eye-tracker API libs
AC_CHECK_LIB_CXX_FULL(eyelink_core, [HAVE_EYELINK], [EYELINK_LIBS],
                      [EyeLink-related programs will be disabled],
                      [#include <math.h>
		       #include <eyelink/eyelink.h>], [eyelink_open()],
                      ,,[-static -leyelink_core_graphics -lsdl_util -lSDL_mixer -lSDL_gfx -lSDL_ttf -lrt -lpthread])

###########################
# OpenCV library

AC_CHECK_LIB_CXX_FULL(opencv_core, [HAVE_OPENCV2], [OPENCV2_LIBS],
                      [OpenCV2-dependent features will be disabled],
                      [#include <cstddef>
                       #include <opencv2/legacy/legacy.hpp>], [CvCalibFilter f; f.SetCameraCount(2);],
                      [CPPFLAGS="$CPPFLAGS -I/usr/local/include"],,[-lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_video ])

if test "x$HAVE_OPENCV2" = "xno"; then
  AC_CHECK_LIB_CXX_FULL(cv, [HAVE_OPENCV1], [OPENCV_LIBS],
                      [OpenCV-dependent features will be disabled],
                      [#include <cstddef>
                       #include <opencv/cv.h>], [cvMatchTemplate(0,0,0,0);],
                      [CPPFLAGS="$CPPFLAGS -I/usr/include/opencv"],,[-lcxcore])
  AC_CHECK_LIB_CXX_FULL(cvaux, [HAVE_OPENCVAUX], [OPENCVAUX_LIBS],
                      [OpenCVaux-dependent features will be disabled],
                      [#include <cstddef>
                       #include <opencv/cvaux.h>], [CvCalibFilter f; f.SetCameraCount(2);],
                      [CPPFLAGS="$CPPFLAGS -I/usr/include/opencv"],,[-lcxcore -lcv -lcvaux])

  AC_CHECK_LIB_CXX_FULL(highgui, [HAVE_HIGHGUI], [HIGHGUI_LIBS],
                      [HighGui-dependent features will be disabled],
                      [#include <opencv/highgui.h>], [cvInitSystem(1,(char**)(&"test"));],
                      [CPPFLAGS="$CPPFLAGS -I/usr/include/opencv"],,[-lcxcore -lcv])
fi

if test "x$HAVE_OPENCV2" = "xyes" || test "x$HAVE_OPENCV1" = "xyes"; then
AC_DEFINE(HAVE_OPENCV,1,[have any OpenCV library?])
fi

# some opencv stuff has been moved to legacy as of version 2.4:
if test "x$HAVE_OPENCV2" = "xyes" && test -f /usr/include/opencv2/legacy/legacy.hpp; then
AC_DEFINE(HAVE_OPENCV_LEGACY,1,[have OpenCV legacy library?])
fi

###########################
# ELECTRIC FENCE
# --enable-efence
# Use Electric Fence if Available

AC_MSG_CHECKING([whether to use Electric Fence Debugging])

AC_ARG_ENABLE(efence,
              [AC_HELP_STRING([--enable-efence],
                             [use Electric Fence for memory debugging])],
              ,
              enable_efence=no)

AC_MSG_RESULT($enable_efence)

# efence sets the LDFLAGS_BIN so it only links into the core INVT binaries
# and not into the compiler binaries such as devscripts/cdeps.cc which
# has a problem with efence. Be sure to use a recent version of efence. There
# is a 64 bit RPM for mandriva with the distro

if test "x$enable_efence" = xyes; then
   AC_LATE_WARN([you have selected to use Electric Fence Debugging])
   LDFLAGS_BIN="$LDFLAGS_BIN -lefence"
fi

###########################
# ODE library
AC_CHECK_LIB_CXX_FULL(ode, [HAVE_ODE], [ODE_LIBS],
                      [associated functionality will be disabled],
                      [#include <ode/ode.h>], [dWorldID dw = dWorldCreate()],
                      ,,[-lX11 -lGL -lGLU])

###########################
# LibSerial library
AC_CHECK_LIB_CXX_FULL(serial, [HAVE_LIBSERIAL], [LIBSERIAL_LIBS],
                      [associated functionality will be disabled],
                      [#include <SerialStream.h>],
                      [LibSerial::SerialStream gogo("/dev/null", std::ios::in)],
                      ,,)

###########################
# libirobot-create library
AC_CHECK_LIB_CXX_FULL(irobot-create, [HAVE_LIBIROBOT_CREATE], [LIBIROBOT_CREATE_LIBS],
                      [associated functionality will be disabled],
                      [#include <irobot-create.hh>],
                      [iRobot::CommandNotAvailable()],
                      ,,)

###########################
# libbluetooth library (bluez-devel)
AC_CHECK_LIB_CXX_FULL(bluetooth, [HAVE_LIBBLUETOOTH], [BLUETOOTH_LIBS],
                      [associated functionality will be disabled],
                      [#include <bluetooth/bluetooth.h>],
                      [baprintf("hello")])

###########################
# libwiimote library
AC_CHECK_LIB_CXX_FULL(cwiimote, [HAVE_LIBWIIMOTE], [WIIMOTE_LIBS],
                      [associated functionality will be disabled],
                      [#include <libcwiimote/wiimote.h>],
                      [wiimote_t wiimote = WIIMOTE_INIT],
                      ,,[-lbluetooth -lm])

###########################
# liblwpr library
AC_CHECK_LIB_CXX_FULL(lwpr, [HAVE_LWPR], [LWPR_LIBS],
                      [associated functionality will be disabled],
                      [#include <lwpr/lwpr.h>],
                      [LWPR_Model model; lwpr_init_model(&model, 2, 1, "test")],
                      ,,[-lm -lpthread -lexpat])

###########################
# xclib CameraLink framegrabber library from Epix Inc
AC_CHECK_LIB_CXX_FULL(xclib_pic, [HAVE_XCLIB], [XCLIB_LIBS],
                      [associated functionality will be disabled],
                      [#include <xclib/xcliball.h>],
                      [struct pximage pi],
                      ,,)

###########################
# libopensurf (hacked version of Chris Evans's OpenSURF library, ECCV '06)
#
# NOTE: This refers to a custom version of Chris Evans's OpenSURF
# library built by Manu Viswanathan specifically for use in iLab. The
# custom version has mostly cosmetic differences from the original such
# as all functions/classes wrappend inside a namespace, code clean-up,
# compiler warnings removed, autoconfiscated library build, etc.
AC_CHECK_LIB_CXX_FULL(opensurf, [INVT_HAVE_LIBSURF], [SURF_LIBS],
                      [SURF-dependent features will be disabled],
                      [#include <opensurf/opensurf.hh>],
                      [opensurf::doSurf(0)],,,)

###########################
# torch library (various machine learning algorithms)
AC_CHECK_LIB_CXX_FULL(torch, [INVT_HAVE_LIBTORCH], [TORCH_LIBS],
                      [torch-dependent features will be disabled],
                      [#include <torch/SVMClassification.h>],
                      [Torch::SVMClassification svm(0)],,,)

###########################
# URG library for interfacing with Hokuyo line of laser range finders
#
# NOTE: This refers to a custom version of Satofumi Kamimura's URG
# library built by Manu Viswanathan specifically for use in iLab. The
# custom version is a plain C API available in a single liburg. The
# original includes C++ classes and spreads the C functionality across
# 3 different libs.
AC_CHECK_LIB_CXX_FULL(urg, [INVT_HAVE_LIBURG], [URG_LIBS],
                      [liburg-dependent features will be disabled],
                      [#include <urg/urg.h>],
                      [urg_connect(0, "", 0)],,,)
###########################
# Freenect library (OpenKinect):
AC_CHECK_LIB_CXX_FULL(freenect, [INVT_HAVE_LIBFREENECT], [FREENECT_LIBS],
                      [libfreenect-dependent features will be disabled],
                      [#include <libfreenect.h>], [freenect_context *f_ctx])

###########################
# OpenNI library (OpenNI):

cppflags_save="$CPPFLAGS"
CPPFLAGS="$CPPFLAGS -isystem /usr/include/openni"
AC_CHECK_LIB_CXX_FULL(OpenNI, [INVT_HAVE_OPENNI], [OPENNI_LIBS],
                      [libOpenNI-dependent features will be disabled],
                       [
// The following is a fix for OpenNI with c++0x
// c++0x does not define "linux" as a preprocessor directive
#ifdef __linux__
#ifndef linux
#define linux 1
#endif // linux
#endif // __linux__
#include<openni/XnOS.h>], [XnStatus rc],
			[CPPFLAGS="$CPPFLAGS -I/usr/include/openni -Wno-unknown-pragmas"],[CPPFLAGS="$cppflags_save"],)

###########################
# NITE library (NITE):
AC_CHECK_LIB_CXX_FULL(XnVNite, [INVT_HAVE_NITE], [NITE_LIBS],
                      [NITE-dependent features will be disabled],
                       [#include<nite/XnVNite.h>], [XnVSessionManager sessionManager],
			[CPPFLAGS="$CPPFLAGS -Wno-unknown-pragmas -I/usr/include/nite"],,)

###########################
# PCL library
CPPFLAGS_SAVE=$CPPFLAGS
# note how we add Werror here to disable PCL if it has warnings...
CPPFLAGS="$CPPFLAGS -I/usr/include/vtk-5.8 -I/usr/include/pcl-1.6 -I/usr/include/eigen3 -I/usr/include/openni -I/usr/include/qhull -Werror"
AC_CHECK_LIB_CXX_FULL(pcl_io, [HAVE_PCL], [PCL_LIBS],
                      [PCL-dependent features will be disabled],
                      [#include <pcl/io/pcd_io.h>
                       #include <pcl/point_types.h>], [pcl::PointCloud<pcl::PointXYZ> cloud;]
                      ,, CPPFLAGS=$CPPFLAGS_SAVE ,
[-lboost_system-mt -lboost_filesystem-mt -lboost_thread-mt -lpthread -lboost_date_time-mt -lboost_iostreams-mt -lpcl_common -lpcl_kdtree -lpcl_octree -lpcl_search -lpcl_sample_consensus -lpcl_filters -lOpenNI /usr/lib/libvtkCommon.so.5.8.0 /usr/lib/libvtkRendering.so.5.8.0 /usr/lib/libvtkHybrid.so.5.8.0 -lpcl_io -lpcl_features -lpcl_keypoints -lpcl_segmentation -lpcl_visualization -lpcl_tracking -lqhull -lpcl_surface -lpcl_registration -lpcl_apps /usr/lib/libvtkParallel.so.5.8.0 /usr/lib/libvtkRendering.so.5.8.0 /usr/lib/libvtkGraphics.so.5.8.0 /usr/lib/libvtkImaging.so.5.8.0 /usr/lib/libvtkIO.so.5.8.0 /usr/lib/libvtkFiltering.so.5.8.0 /usr/lib/libvtkCommon.so.5.8.0 -lm /usr/lib/libvtksys.so.5.8.0 -ldl])

###########################
# Bumblebee2 Stereo Camera library
AC_CHECK_LIB_CXX_FULL(pgrlibdcstereo, [HAVE_BUMBLEBEE2], [BUMBLEBEE2_LIBS],
                      [Bumblebee2 Stereo Camera libaray dependent features will be disabled],
                      [ #include <pgr_registers.h>
                        #include <pgr_stereocam.h> ], [PGRStereoCamera_t stereoCamera;]
                      ,,, [-lpgrlibdcstereo -ltriclops -lpnmutils])

###########################

# build most objects as shared libs?
if test "x$enable_shlibs" = xyes; then
   CXXFLAGS="$CXXFLAGS -fpic"
   CFLAGS="$CFLAGS -fpic"
   o="so"
fi

AC_SUBST(o)

AC_MSG_CHECKING([how to time build commands])

# check if /usr/bin/time accepts a fancy --format option

(/usr/bin/time --format='<<time>> %U user, %S sys, %E elapsed, %P CPU' true) > /dev/null 2>&1
status=$?
if test $status -eq 0; then
  # Command that can be used in a makefile to generate a time report
  # that includes the name of the target ($@); this is especially useful
  # in parallel builds, where the time reports are interleaved with the
  # build commands and so it's impossible to determine which are paired
  # with which. The lines all start with a "<<time>>" prefix, so it's easy
  # to grep all of the time lines out of a build log.
  TIME="/usr/bin/time --format='<<time>> %U user, %S sys, %E elapsed, %P CPU : \$@'"
else
  TIME="time"
fi

AC_MSG_RESULT([$TIME])
AC_SUBST(TIME)

# as of 2007-02-07, we have no java or JNI code in the src tree any
# more (IRoom and agentMinder were obsoleted), so we have no need for
# any java configuration; still, let's just disable the configuration
# code here rather than removing it entirely, in case we need to
# re-enable it in the future

dnl    # check for a Sun java sdk
dnl    JAVAPATH=/usr/java/j2sdk/bin:/usr/java/j2sdk1.4.2_05/bin:/usr/java/jdk1.5.0/bin:/usr/java/jdk1.5.0_01/bin:/usr/lib/jvm/java-1.5.0-sun/bin:/usr/java/j2sdk1.4.2_12/bin:/usr/java/jdk1.5.0_09/bin
dnl    AC_PATH_PROG(JAVA, [java], , [$JAVAPATH])
dnl    if test "x$JAVA" = "x"; then
dnl      AC_LATE_WARN([cannot find 'java' program])
dnl    else
dnl      AC_PATH_PROG(JAVAC, [javac], , [$JAVAPATH])
dnl      if test "x$JAVAC" = "x"; then
dnl        AC_LATE_WARN([cannot find 'javac' program])
dnl      fi
dnl
dnl      JAVAROOT=$(dirname $(dirname $JAVA))
dnl      if test -d $JAVAROOT/include; then
dnl        CPPFLAGS="-I$JAVAROOT/include $CPPFLAGS"
dnl      fi
dnl      if test -d $JAVAROOT/include/linux; then
dnl        CPPFLAGS="-I$JAVAROOT/include/linux $CPPFLAGS"
dnl      fi
dnl
dnl      AC_CHECK_HEADERS([jni.h])
dnl    fi

# export our final list of possible include paths and final CPPFLAGS:
if test "x$ADDINC" != "x"; then
  CPPFLAGS="$CPPFLAGS $ADDINC"
fi
AC_SUBST(ADDINC)

##########################################################################
### Checks for typedefs, structures.

AC_DEFINE_IF_COMPILE([INT_IS_32_BITS],
                     [whether int is a 32-bit type],
                     [typedef char arr[[sizeof(int) == 4 ? 1 : -1]];],
                     [arr a;])

AC_DEFINE_IF_COMPILE([LONG_IS_32_BITS],
                     [whether long is a 32-bit type],
                     [typedef char arr[[sizeof(long) == 4 ? 1 : -1]];],
                     [arr a;])

AC_DEFINE_IF_COMPILE([LONG_IS_64_BITS],
                     [whether long is a 64-bit type],
                     [typedef char arr[[sizeof(long) == 8 ? 1 : -1]];],
                     [arr a;])

AC_DEFINE_IF_COMPILE([LONG_LONG_IS_64_BITS],
                     [whether long long is a 64-bit type],
                     [typedef char arr[[sizeof(long long) == 8 ? 1 : -1]];],
                     [arr a;])

AC_DEFINE_IF_COMPILE([SIZE_T_IS_UNSIGNED_INT],
                     [whether size_t is a typedef for unsigned int],
                     [#include <cstdlib>],
                     [extern void foo(unsigned int*);
                      size_t i;
                      foo(&i);])

AC_DEFINE_IF_COMPILE([SIZE_T_IS_UNSIGNED_LONG],
                     [whether size_t is a typedef for unsigned long],
                     [#include <cstdlib>],
                     [extern void foo(unsigned long*);
                      size_t i;
                      foo(&i);])

AC_CHECK_TYPES(sighandler_t,,,[#include <signal.h>])

##########################################################################
### Checks for library functions.


### Check for missing iostream instantiation in the c++ std lib
AC_DEFINE_IF_NOT_LINK([GVX_MISSING_IOSTREAM_INSTANTIATION],
                      [whether libstdc++ is lacking a full std::iostream instantiation],
                      [#include <istream>],
                      [class stdiostream : public std::iostream
                       {
                         public:
                             stdiostream() : std::iostream(0) {}
                       };

                       stdiostream s;])

### Check whether libc supports the "%zu" printf() extension for size_t
AC_DEFINE_IF_RUN([HAVE_PRINTF_Z_MODIFIER],
                 [whether printf() allows "%zu" for size_t arguments],
                 [#include <cstdio>],
                 [[size_t i=3;
                   char buf[16];
                   buf[0] = '\0';
                   sprintf(buf, "%zu", i);
                   if (buf[0] == '3' && buf[1] == '\0') exit(0);
                   exit(1);]])

### Check whether posix semaphores are present and working
# (on Mac OSX, sem_init() gives errno=78, "Function not implemented")
LIBS_SAVE=$LIBS
LIBS="$LIBS -pthread"
AC_DEFINE_IF_RUN([HAVE_WORKING_POSIX_SEMAPHORES],
                 [whether posix semaphores are present and working properly],
                 [#include <stdlib.h>
                  #include <semaphore.h>],
                 [[sem_t semaphore;
                   int ret = sem_init(&semaphore, 0, 0);
                   if (ret == 0) exit(0);
                   exit(1);]])
LIBS=$LIBS_SAVE

# Note: we put a double underscore at the start of the key so that
# this chunk gets sorted to the very end of config.h.in; that way it's
# sure to be AFTER the definitions of SIZE_T_IS_UNSIGNED_INT and
# SIZE_T_IS_UNSIGNED_LONG.

AH_VERBATIM([__ZU],
[/* printf conversion specifier to use for size_t */
#if defined(HAVE_PRINTF_Z_MODIFIER)
#  define ZU "zu"
#elif defined(SIZE_T_IS_UNSIGNED_INT)
#  define ZU "u"
#elif defined(SIZE_T_IS_UNSIGNED_LONG)
#  define ZU "lu"
#else
#  error "don't know which conversion specifier to use for size_t for printf()"
#endif])

##########################################################################
### Generate output

CPPFLAGS="$CPPFLAGS -include config.h"

if test "x$enable_debug" = xyes; then
   CXXFLAGS="-g $CXXFLAGS"
   CFLAGS="-g $CFLAGS"
fi

# finally prepend CXXWARNINGS to CXXFLAGS (this must be at the very
# end of the configure script so that flags requesting compiler
# warnings don't mess up earlier configuration tests)
CXXFLAGS="$CXXWARNINGS $CXXFLAGS"
CFLAGS="$CWARNINGS $CFLAGS"

# loop over our LDFLAGS so we can compute some auxiliary variables --
# possibly add -rpath linker options, and build a VPATH_LIBDIRS
# variable that can be used in the Makefile
LDFLAGS_SAVE=$LDFLAGS
VPATH_LIBDIRS=""
for opt in $LDFLAGS_SAVE; do
   case $opt in
      -L*)
         # trim the leading '-L' and replace any double slashes with
         # single slashes
         dir=`echo $opt | cut -c 3- | sed 's,//,/,g'`

         if test "x$VPATH_LIBDIRS" = "x"; then
            VPATH_LIBDIRS=$dir
         else
            VPATH_LIBDIRS="${VPATH_LIBDIRS}:$dir"
         fi

         # If the linker will accept a -rpath option, then generate
         # -Wl,-rpath options in LDFLAGS for each of the -L
         # directories in the library search path; that way, the same
         # libraries will be found at runtime without having to set
         # the LD_LIBRARY_PATH environment variable
         if test "x$linker_accepts_rpath" = "xyes"; then
            case $dir in
               /usr/lib|/usr/lib64)
                  # don't put /usr/lib or /usr/lib64 in rpath since
                  # that will unnecessarily override a user's attempt
                  # to pull in custom libraries via an LD_LIBRARY_PATH
                  # environment variable
                  ;;
               *)
                  LDFLAGS="$LDFLAGS -Wl,-rpath,$dir"
                  ;;
            esac
         fi
         ;;
   esac
done

AC_SUBST(LDFLAGS)
AC_SUBST(LDFLAGS_BIN)
AC_SUBST(VPATH_LIBDIRS)

AC_MSG_NOTICE([VPATH_LIBDIRS=$VPATH_LIBDIRS])

AC_REPORT_FEATURE_STATUS()
AC_GIVE_LATE_WARNINGS()

AH_TOP([#ifndef INVT_CONFIG_H_200604201237_DEFINED
#define INVT_CONFIG_H_200604201237_DEFINED])

AH_BOTTOM([#endif /* INVT_CONFIG_H_200604201237_DEFINED */])

rm depoptions-all.in
for f in `find -L . -name depoptions.in -print`; do
  DEPFILES="$DEPFILES $f"
  echo "Using: " $f
  cat $f >> depoptions-all.in
done
AC_SUBST(DEPFILES)

AC_CONFIG_FILES([Makefile:Makefile.in])
AC_CONFIG_FILES([mexopts.sh:mexopts.sh.in])
AC_CONFIG_FILES([depoptions-all:depoptions-all.in])
AC_CONFIG_FILES([tests/config.tcl:tests/config.tcl.in])
AC_CONFIG_FILES([tests/invt_config.pm:tests/invt_config.pm.in])
AC_CONFIG_HEADERS([config.h])
AC_OUTPUT()
