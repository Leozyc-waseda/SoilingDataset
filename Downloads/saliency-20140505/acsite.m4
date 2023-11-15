######################### -*- autoconf -*- ####################################
# acsite.m4
#
# An m4 macro file for configure.ac
#
###############################################################################



###############################################################################
#
# Imported from file "gwqt.m4" in autoqt-0.03
#
# This macro was written originally by Geoffrey Wossum, and subsequently
# modified by Allan Clark and Murray Cumming.  Please see the autoqt site at
# http://autoqt.sourceforge.net/ for the latest version of this script.
#
# This script was modified by Zhan Shi for use with the iLab C++ Neuromorphic
# Vision Toolkit.
#
#
# Copyright (c) 2002, Geoffrey Wossum
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# - Neither the name of Geoffrey Wossum nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

# Check for Qt3 compiler flags, linker flags, and binary packages
AC_DEFUN([gw_CHECK_QT3],
[
AC_REQUIRE([AC_PROG_CXX])
AC_REQUIRE([AC_PATH_X])

AC_ARG_VAR([QT3DIR], [root of Qt3 installation])

AC_MSG_CHECKING([QT3DIR])
AC_ARG_WITH([qt3dir],
            [AC_HELP_STRING([--with-qt3dir=DIR],
                            [Qt3 installation directory [default=$QT3DIR]])])

# this will eventually point to the directory that contains qglobal.h
QT3_INC_DIR=""

case "x$with_qt3dir" in
    xno)
        # the user specifically requested no qt (either with
        # --with-qt3dir=no or --without-qt3dir), so let's do that:
        QT3DIR=""
        ;;

    x)
        # the user didn't give --with-qt3dir or --without-qt3dir, so
        # let's try to pick up the QT3DIR environment variable, or else
        # use a sensible default:
        if test "x$QT3DIR" = x ; then
            for d in qt qt2 qt3 qt31 qt-3.1 qt32 qt-3.2 qt33 qt-3.3; do
                for i in /usr/local/$d /usr/lib/$d /usr/lib64/$d; do
                    if test -f $i/include/qglobal.h; then
                        QT3DIR=$i
                        QT3_INC_DIR="$i/include"
                    fi
                done

                if test -f /usr/include/$d/qglobal.h; then
                    QT3DIR=/usr
                    QT3_INC_DIR="/usr/include/$d"
                fi
            done
	else
            if test -f $QT3DIR/include/qglobal.h; then
                QT3_INC_DIR="$QT3DIR/include"
            fi
        fi
        ;;

    *)
        # the user gave a --with-qt3dir=value, so let's use that value
        QT3DIR=$with_qt3dir
	QT3_INC_DIR="$QT3DIR/include"
        ;;
esac

if test x"$QT3DIR" = x ; then
    AC_MSG_RESULT([missing])
    AC_LATE_WARN([Qt3 executables will not be built (QT3DIR must be defined, or --with-qt3dir option given)])
    AC_SUBST(MOC3, true)
else

    AC_MSG_RESULT([$QT3DIR])

    AC_DEFINE(INVT_HAVE_QT3,1,[build with Qt3 support?])

    # Figure out which version of Qt3 we are using
    AC_MSG_CHECKING([Qt3 version])
    if test -f "$QT3_INC_DIR/qglobal.h"; then
        # ok, we already figured out $QT3_INC_DIR above
        :
    elif test -f $QT3DIR/include/qglobal.h; then
        QT3_INC_DIR="$QT3DIR/include"
    elif test -f $QT3DIR/include/qt/qglobal.h; then
        QT3_INC_DIR="$QT3DIR/include/qt"
    elif test -f $QTDIR/include/qt3/qglobal.h; then
        QT3_INC_DIR="$QT3DIR/include/qt3"
    elif test -f $(dirname $(dirname $QT3DIR))/include/`basename $QT3DIR`/qglobal.h; then
        QT3_INC_DIR=$(dirname $(dirname $QT3DIR))/include/`basename $QT3DIR`
    fi

    if test "x$QT3_INC_DIR" != "x"; then
        QT3_VER=$(expr "`grep 'define.*QT_VERSION_STR\W' $QT3_INC_DIR/qglobal.h`" : '.*\"\(.*\)\"')
        QT3_MAJOR=$(expr "$QT3_VER" : '\([[0-9]][[0-9]]*\)')
        QT3_MINOR=$(expr "$QT3_VER" : '[[0-9]][[0-9]]*'.'\([[0-9]][[0-9]]*\)')
        CPPFLAGS="$CPPFLAGS -I$QT3_INC_DIR"
        AC_MSG_RESULT([$QT3_VER (major version $QT3_MAJOR, minor version $QT3_MINOR) in $QT3_INC_DIR/qglobal.h])
        AC_DEFINE_UNQUOTED(INVT_QT3_MINOR, ${QT3_MINOR}, [Qt3 Minor Version Number])
    else
        AC_MSG_RESULT([unknown (no such file: $QT3DIR/include/qglobal.h)])
    fi

    # Check that moc is in path
    AC_PATH_PROGS(MOC3, [moc-qt3 moc], , $QT3DIR/bin:$PATH)
    if test x$MOC3 = x ; then
        AC_MSG_ERROR([couldn't find Qt3 moc in $QT3DIR/bin:$PATH])
    fi

    # uic is the Qt user interface compiler
    AC_PATH_PROGS(UIC3, [uic-qt3 uic], , $QT3DIR/bin:$PATH)
    if test x$UIC3 = x ; then
        AC_MSG_ERROR([couldn't find Qt3 uic in $QT3DIR/bin:$PATH])
    fi

    # Check for a multithreaded Qt library
    if test "x`ls $QT3DIR/lib64/libqt-mt.* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt-mt"
        QT3_IS_MT="yes"
        LDFLAGS="$LDFLAGS -L$QT3DIR/lib64"
    elif test "x`ls $QT3DIR/lib64/libqt.* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt"
        QT3_IS_MT="no"
        LDFLAGS="$LDFLAGS -L$QT3DIR/lib64"
    elif test "x`ls $QT3DIR/lib/libqt-mt.* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt-mt"
        QT3_IS_MT="yes"
        LDFLAGS="$LDFLAGS -L$QT3DIR/lib"
    elif test "x`ls $QT3DIR/lib/libqt.* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt"
        QT3_IS_MT="no"
        LDFLAGS="$LDFLAGS -L$QT3DIR/lib"
    elif test "x`ls /usr/lib/libqt-mt* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt-mt"
        QT3_IS_MT="yes"
        LDFLAGS="$LDFLAGS -L/usr/lib"
    elif test "x`ls /usr/lib64/libqt-mt* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt-mt"
        QT3_IS_MT="yes"
        LDFLAGS="$LDFLAGS -L/usr/lib64"
    elif test "x`ls /usr/lib64/libqt* 2> /dev/null`" != x ; then
        QT3_LIBS="-lqt"
        QT3_IS_MT="no"
        LDFLAGS="$LDFLAGS -L/usr/lib64"
    else
        AC_MSG_ERROR([couldn't find any Qt3 libraries in $QT3DIR/lib])
    fi

    # we require a multithreaded Qt3 library because we use
    # QApplication::lock()
    if test x"$QT3_IS_MT" = "xyes" ; then
        # note: we define QT_THREAD_SUPPOR as opposed to QT3_THREAD_SUPPORT because the
        # QT_THREAD_SUPPORT variable is used by the QT3 include files:
        AC_DEFINE(QT_THREAD_SUPPORT,1,[use multi-threaded Qt3 library?])
    else
        AC_MSG_ERROR([couldn't find multithreaded Qt3 libraries in $QT3DIR/lib])
    fi

    AC_MSG_CHECKING([if Qt3 is multithreaded])
    AC_MSG_RESULT([$QT3_IS_MT])

    AC_SUBST(QT3DIR)
    AC_SUBST(QT3_LIBS)
fi

])

##############################################################################
# Check for Qt4 compiler flags, linker flags, and binary packages
AC_DEFUN([gw_CHECK_QT4],
[
AC_REQUIRE([AC_PROG_CXX])
AC_REQUIRE([AC_PATH_X])

AC_ARG_VAR([QT4DIR], [root of Qt4 installation])

AC_MSG_CHECKING([QT4DIR])
AC_ARG_WITH([qt4dir],
            [AC_HELP_STRING([--with-qt4dir=DIR],
                            [Qt4 installation directory [default=$QT4DIR]])])

# this will eventually point to the directory that contains qglobal.h
QT4_INC_DIR=""

case "x$with_qt4dir" in
    xno)
        # the user specifically requested no qt (either with
        # --with-qt4dir=no or --without-qt4dir), so let's do that:
        QT4DIR=""
        ;;

    x)
        # the user didn't give --with-qt4dir or --without-qt4dir, so
        # let's try to pick up the QT4DIR environment variable, or else
        # use a sensible default:
        if test "x$QT4DIR" = x ; then
            for d in qt4; do
                for i in /usr/local/$d /usr/lib/$d /usr/lib64/$d; do
                    if test -f $i/include/Qt/qglobal.h; then
                        QT4DIR=$i
                        QT4_INC_DIR="$i/include"
                    fi
                done

                if test -f /usr/include/$d/Qt/qglobal.h; then
                    QT4DIR=/usr
                    QT4_INC_DIR="/usr/include/$d"
                fi

            done
	else
            if test -f $QT4DIR/include/Qt/qglobal.h; then
                QT4_INC_DIR="$QT4DIR/include"
            fi
        fi
        ;;

    *)
        # the user gave a --with-qt4dir=value, so let's use that value
        QT4DIR=$with_qt4dir
	QT4_INC_DIR="$QT4DIR/include"
        ;;
esac

if test x"$QT4DIR" = x ; then
    AC_MSG_RESULT([missing])
    AC_LATE_WARN([Qt4 executables will not be built (QT4DIR must be defined, or --with-qt4dir option given)])
    AC_SUBST(MOC4, true)
else

    AC_MSG_RESULT([$QT4DIR])

    AC_DEFINE(INVT_HAVE_QT4,1,[build with Qt4 support?])

    # Figure out which version of Qt4 we are using
    AC_MSG_CHECKING([Qt4 version])

    if test "x$QT4_INC_DIR" != "x"; then
        QT4_VER=$(expr "`grep 'define.*QT_VERSION_STR\W' $QT4_INC_DIR/Qt/qglobal.h`" : '.*\"\(.*\)\"')
        QT4_MAJOR=$(expr "$QT4_VER" : '\([[0-9]][[0-9]]*\)')
        QT4_MINOR=$(expr "$QT4_VER" : '[[0-9]][[0-9]]*'.'\([[0-9]][[0-9]]*\)')
        CPPFLAGS="$CPPFLAGS -I$QT4_INC_DIR"
        AC_MSG_RESULT([$QT4_VER (major version $QT4_MAJOR, minor version $QT4_MINOR) in $QT4_INC_DIR/Qt/qglobal.h])
        AC_DEFINE_UNQUOTED(INVT_QT4_MINOR, ${QT4_MINOR}, [Qt4 Minor Version Number])
    else
        AC_MSG_RESULT([unknown (no such file: $QT4DIR/include/Qt/qglobal.h)])
    fi

    # Check that moc is in path
    AC_PATH_PROGS(MOC4, [moc-qt4 moc], , $QT4DIR/bin:$PATH)
    if test x$MOC4 = x ; then
        AC_MSG_ERROR([couldn't find Qt4 moc in $QT4DIR/bin:$PATH])
    fi

    # uic is the Qt user interface compiler
    AC_PATH_PROGS(UIC4, [uic-qt4 uic], , $QT4DIR/bin:$PATH)
    if test x$UIC4 = x ; then
        AC_MSG_ERROR([couldn't find Qt4 uic in $QT4DIR/bin:$PATH])
    fi

    # Check for Qt library. FIXME: should allow to use 32-bit libs even if the 64-bit is installed
    if test "x`ls $QT4DIR/lib64/libQt* 2> /dev/null`" != x ; then
        QT4_LIBS="-lQtGui -lQtCore"
        LDFLAGS="$LDFLAGS -L$QT4DIR/lib64"
    elif test "x`ls $QT4DIR/lib/libQt* 2> /dev/null`" != x ; then
        QT4_LIBS="-lQtGui -lQtCore"
        LDFLAGS="$LDFLAGS -L$QT4DIR/lib"
    elif test "x`ls /usr/lib64/libQt* 2> /dev/null`" != x ; then
        QT4_LIBS="-lQtGui -lQtCore"
        LDFLAGS="$LDFLAGS -L/usr/lib64"
    elif test "x`ls /usr/lib/libQt* 2> /dev/null`" != x ; then
        QT4_LIBS="-lQtGui -lQtCore"
        LDFLAGS="$LDFLAGS -L/usr/lib"
    elif test "x`ls /usr/lib/x86_64-linux-gnu/libQt* 2> /dev/null`" != x ; then
        QT4_LIBS="-lQtGui -lQtCore"
        LDFLAGS="$LDFLAGS -L/usr/lib/x86_64-linux-gnu"
    else
        AC_MSG_ERROR([couldn't find any Qt4 libraries in $QT4DIR/lib])
    fi

    AC_SUBST(QT4DIR)
    AC_SUBST(QT4_LIBS)
fi

])

############################End of autoqt macro script#########################
