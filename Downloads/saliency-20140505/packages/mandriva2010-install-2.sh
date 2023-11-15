#!/bin/sh

# do an svn checkout of the saliency package, then

# Run this script as root from within the saliency/ directory

if [ ! -f configure.ac ]; then
    echo "You must run this script AS ROOT from within the saliency/ directory -- ABORT"
    exit 1;
fi

# setup urpmi sources?
read -p "Do you want to set up urpmi media sources (Y/n)?"
if [ "$REPLY" == "y" ]; then
    urpmi.addmedia --distrib --mirrorlist http://api.mandriva.com/mirrors/basic.2010.0.x86_64.list
    urpmi.addmedia --distrib --mirrorlist http://plf.zarb.org/mirrors/2010.0.x86_64.list
fi

# drakconf remove cd source and add main, updates, mirror list, and all the backport
read -p "Do you want to setup package source for urpmi (Y/n)?"
if [ "$REPLY" == "y" ]; then
    # first, remove all the ignore statement, we will add a few select ones back later:
    sed --in-place=.bak "/ignore/d" /etc/urpmi/urpmi.cfg

    # disable CD-ROM sources:
    sed --in-place=.bak2 "/cdrom:/a\ \ ignore" /etc/urpmi/urpmi.cfg

    # disable debug sources:
    sed --in-place=.bak3 "/\ debug/a\ \ ignore" /etc/urpmi/urpmi.cfg
    sed --in-place=.bak4 "/^debug/a\ \ ignore" /etc/urpmi/urpmi.cfg

    # disable 32-bit sources:
    sed --in-place=.bak5 "/32/a\ \ ignore" /etc/urpmi/urpmi.cfg

    # disable testing sources:
    sed --in-place=.bak6 "/Testing/a\ \ ignore" /etc/urpmi/urpmi.cfg
fi

# run urpmi update?
read -p "Do you want to run urpmi.update (Y/n)?"
if [ "$REPLY" == "y" ]; then
    urpmi.update -a
fi

### first the packages that require a manual answer:
# pick one, as they conflict...
urpmi --force lib64dc1394_12
urpmi --force lib64dc1394_12-devel
#urpmi --force dc1394_22-devel

urpmi --force qt3-devel

# replace line 61 in /usr/lib/qt3/include/qimage.h  
#  { return key < other.key || (key==other.key && lang < other.lang); }
read -p "Do you want to fix qimage.h 61 problem (Y/n)?"
if [ "$REPLY" == "y" ]; then
    cat /usr/lib/qt3/include/qimage.h | \
	perl -pe 's/.*/  { return key < other.key || (key==other.key && lang < other.lang); }/ if $. == 61' > ~/tmp.txt
    cp -f ~/tmp.txt /usr/lib/qt3/include/qimage.h
    rm -f ~/tmp.txt
fi

# these will be needed, can maybe avoid some question by installing them all upfront:
urpmi --force lib64jpeg62
urpmi --force lib64jpeg7
urpmi --force lib64jpeg-devel-7

urpmi --force make
urpmi --force autoconf-archive   # for lots of aclocal autoconf rules
urpmi --force emacs
urpmi --force tcl-devel
urpmi --force tk-devel
urpmi --force popt-devel
urpmi --force bzip2-devel
urpmi --force ffmpeg-devel
urpmi --force fftw-devel
urpmi --force gsl-devel
urpmi --force blas-devel
urpmi --force lapack-devel
urpmi --force avc1394-devel
urpmi --force raw1394-utils
urpmi --force SDL-devel
urpmi --force SDL_gfx-devel
urpmi --force SDL_mixer-devel
urpmi --force SDL_image-devel
urpmi --force SDL_ttf-devel
urpmi --force openmpi
urpmi --force gd-devel
urpmi --force readline-devel
urpmi --force termcap-devel
urpmi --force ode-devel
urpmi --force bluez-devel
urpmi --force cmake
urpmi --force opencv-devel
urpmi --force opencv-samples
./packages/hacks/patch-opencv-path-bug.sh
/bin/cp ./packages/hacks/cvconfig.h /usr/include/opencv/

# liveMedia:
urpmi --force live
urpmi --force live-devel
/bin/ln -s /usr/lib64/live/BasicUsageEnvironment/libBasicUsageEnvironment.a /usr/lib64/libBasicUsageEnvironment.a
/bin/ln -s /usr/lib64/live/groupsock/libgroupsock.a /usr/lib64/libgroupsock.a
/bin/ln -s /usr/lib64/live/liveMedia/libliveMedia.a /usr/lib64/libliveMedia.a
/bin/ln -s /usr/lib64/live/UsageEnvironment/libUsageEnvironment.a /usr/lib64/libUsageEnvironment.a
R='./packages/hacks/replace'
for f in BasicUsageEnvironment groupsock liveMedia UsageEnvironment; do
    /bin/mkdir -p /usr/include/$f
    /bin/cp /usr/lib64/live/$f/include/* /usr/include/$f/
done
for f in BasicUsageEnvironment groupsock liveMedia UsageEnvironment; do
    for x in `/bin/ls /usr/include/$f`; do
	for z in BasicUsageEnvironment groupsock liveMedia UsageEnvironment; do
	    $R "#include <$x>" "#include <$f/$x>" -- /usr/include/$z/*
	    $R "#include \"$x\"" "#include <$f/$x>" -- /usr/include/$z/*
	done
    done
done

# and don't forget the essentials:
urpmi --force xv
urpmi --force mencoder
urpmi --force mplayer
urpmi --force zsh
urpmi --force qt4-designer
urpmi --force qt4-database-plugin-pgsql

# needed for ice 3.3.0 (in http://ilab.usc.edu/packages/forall/current/)
urpmi --force db4.6
urpmi --force libdbcxx4.6
urpmi --force openssl #(should be installed already)
urpmi --force expat #(should be installed already)

# for python ssh tools
urpmi --force python-paramiko
urpmi --force python-vte

# qwt
urpmi --force qwt-devel
/bin/mkdir /usr/include/qwt
/bin/mv /usr/include/qwt* /usr/include/qwt/

# needed for lwpr:
urpmi --force expat-devel

# for cuda:
urpmi --force nvidia-cuda-toolkit
urpmi --force nvidia-cuda-toolkit-devel
urpmi --force nvidia-cuda-profiler
urpmi --force glew
urpmi --force libglew-devel

chkconfig nvidia on
service nvidia start

# needed by matlab R2010a:
urpmi --force lib64xp6

# this is useful to extract metadata from a variety of files:
urpmi --force libextractor

# boost:
urpmi --force boost-devel

# postgres devel:
urpmi --force lib64openssl0.9.8-devel
urpmi --force lib64pq8.4_5-8.4.2
urpmi --force postgresql8.4-devel-8.4.2
urpmi --force libpqxx-devel

######################################################################
# the following is for iLab machines only:
######################################################################

if [ ! -d /lab/packages/forall ]; then
    echo "Skipping additional packages which are available on iLab machines only"
    exit 2
fi

# local packages
urpmi gnuplot
cd /lab/packages/forall/current
rpm -Uvh libserial*
rpm -Uvh --nodeps libirobot*
rpm -Uvh libsvm*
rpm -Uvh libtorch*
rpm -Uvh sparsehash*
rpm -Uvh libopensurf*
rpm -Uvh liburg*
rpm -Uvh lib64cwiimote*

rpm -Uvh lwpr*
for x in `/bin/ls /usr/include/lwpr`; do
    $R "#include <$x>" "#include <lwpr/$x>" -- /usr/include/lwpr/*
    $R "#include \"$x\"" "#include <lwpr/$x>" -- /usr/include/lwpr/*
done

rpm -Uvh ice-3.3.1-1.rhel5.noarch.rpm

# In the command below, first run it without the --nodeps and make
# sure you only get complaints about libcrypto, libssl and libdb,
# all of which should be installed already, then
rpm -Uvh --nodeps ice-c++-devel-3.3.1-1.rhel5.x86_64.rpm \
  ice-libs-3.3.1-1.rhel5.x86_64.rpm \
  ice-servers-3.3.1-1.rhel5.x86_64.rpm \
  ice-utils-3.3.1-1.rhel5.x86_64.rpm

# cuda thrust library
cd /usr/local
unzip /lab/packages/forall/current/thrust-v1.2.zip
ln -s /usr/local/thrust /usr/include/thrust

# commercial packages:
cd /
tar jxvf /lab/vault/software/EyeLink-1000/eyelink-1000-installed.tbz
tar jxvf /lab/vault/software/epix/xclib-installed.tbz


######################################################################
# get rid of orphans:
######################################################################

urpme --auto-orphans
