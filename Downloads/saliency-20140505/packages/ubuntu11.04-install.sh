#!/bin/bash

# Installation guide for Ubuntu 10.04LTS Desktop amd64

# RUN THIS SCRIPT AS ROOT

# Before running this script follow the instructions here:

# http://ilab.usc.edu/toolkit/documentation-ubuntu.shtml

# and, during installation, you should:

# - setup partitions as:
#   / (primary) [typically 10-20GB]
#   swap (primary) [typically 1-2x your RAM size)
#   /home (primary) [remainder of your disk]

# If you are upgrading, make sure that you do not erase the contents of /home!

# - When asked for your name, use "Administrator Account", username
#   "administrator". Do not use your real name since your user account
#   will be handled through NIS.

# - If you have a dual ethernet card, the network will not configure
#   properly. Even with a single ethernet, by default it will use DHCP
#   which is not good. After installation is complete, reboot and
#   login using the administrator account. Then go to System ->
#   Preferences -> Network Connections, and:

# - configure your network (IPv4):
#    - ask Laurent for your IP address, or run 'ypcat -k hosts | grep YOURHOST' on 
#      an ilab machine, where YOURHOST is the hostname of your box
#    - netmask is 255.255.255.0
#    - gateway is 192.168.0.100
#    - DNS servers: 128.125.7.23, 128.125.253.194, 128.125.253.143

# setup software sources:
read -p "Do you want to setup software sources (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    # to setup sources, add them to /etc/apt/sources.list
    /bin/mv /etc/apt/sources.list /etc/apt/sources.list.bak
    /bin/cat > /etc/apt/sources.list <<EOF
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ natty main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ natty-updates main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ natty-backports main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ natty-security main restricted universe multiverse
deb http://archive.canonical.com/ubuntu natty partner
EOF
    apt-get update
fi

# install some essentials:
read -p "Do you want to install emacs, subversion, sshd (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
  apt-get -y install emacs
  apt-get -y install subversion
  apt-get -y install openssh-server
fi

# setup nis client:
read -p "Do you want setup the NIS client (enter 'lab' when asked for domain name) (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    echo "Setup NIS server"
    echo "Editing.. /etc/defaultdomain"
    echo "lab" > /etc/defaultdomain
    apt-get -y install nis
    apt-get -y install autofs
    echo "Editing.. /etc/yp.conf"
    echo "domain lab server 192.168.0.200" >> /etc/yp.conf
    echo "domain lab server 192.168.0.197" >> /etc/yp.conf
    echo "Editing.. /etc/nsswitch.conf"
    mv /etc/nsswitch.conf /etc/nsswitch.conf.org
    cat > /etc/nsswitch.conf <<EOF
passwd:         files nis compat
shadow:         files nis compat
group:          files nis compat
hosts:          files nis mdns4_minimal [NOTFOUND=return] dns mdns4
networks:       files nis
services:       db files nis
protocols:      db files nis
rpc:            db files nis
ethers:         db files nis
netmasks:       files nis
netgroup:       files nis
publickey:      files nis
bootparams:     files nis
automount:      files nis
aliases:        files nis
EOF
    nisdomainname lab
    /etc/init.d/nis restart
    /etc/init.d/autofs restart
    update-rc.d nis enable
    update-rc.d autofs enable
fi

# move /usr/local to /home/local:
read -p "Do you want to delete /usr/local and make it point to /home/local (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    if [ -d /home/local ]; then # already exists
	/bin/rm -rf /usr/local
	ln -s /home/local /usr/local
    else
	/bin/mv /usr/local /home/
	ln -s /home/local /usr/local
    fi
fi

# setup exports:
read -p "Do you want to setup exports (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    apt-get -y install nfs-kernel-server
    for f in /home*/*; do
	bf=`basename $f`
	if [ -d $f -a "x$bf" != "xlost+found" ]; then
	    echo "$f 192.168.0.0/24(rw,async,no_subtree_check) 127.0.0.0/16(rw,async,no_subtree_check)" >> /etc/exports
	fi
    done
    exportfs -r
fi

# add a toor account?
read -p "Do you want to add a toor account (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    echo 'toor:x:0:0:Toor Account:/root:/bin/bash' >> /etc/passwd
    chmod u+w /etc/shadow
    echo 'toor:$6$hZrTpviz$mQaXcS4Lxv/Z/hNwaxkBP8wEYuzlCR3BMI6xQzEy7Zgq0JvQD0dQO2Ay2B8VdwAUWT9NZoYGX7nYYjzRnpb6l/:14818:0:99999:7:::' >> /etc/shadow
    chmod u-w /etc/shadow
fi

# turn off dm:
read -p "Finally, do you want to turn off gdm (will take effect after reboot) (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    sed --in-place=bak s/016/0126/ /etc/init/gdm.conf
    update-rc.d -f gdm remove
fi

# install packages needed to compile ilab software:
read -p "Do you want to install Linux tools and libraries needed to compile iLab code (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-get update

    apt-get -y install build-essential
    apt-get -y install autoconf
    apt-get -y install autoconf-archive
    apt-get -y install automake

    apt-get -y install vim-gtk
    apt-get -y install tcl-dev
    apt-get -y install tk-dev
    apt-get -y install libpopt-dev
    apt-get -y install bzip2 libbz2-dev
    apt-get -y install ffmpeg
    apt-get -y install fftw3-dev
    apt-get -y install gsl
    apt-get -y install gdb
    apt-get -y install libgsl0-dev
    apt-get -y install libblas-dev
    #apt-get -y install lapack-dev
    apt-get -y install libavc1394-dev

    apt-get -y install libdc1394-22-dev
    apt-get -y install libraw1394-dev
    apt-get -y install libsdl1.2-dev
    apt-get -y install libsdl-gfx1.2-dev
    apt-get -y install libsdl-mixer1.2-dev
    apt-get -y install libsdl-image1.2-dev
    apt-get -y install libsdl-ttf2.0-dev
    apt-get -y install libopenmpi-dev
    apt-get -y install libgd-tools
    apt-get -y install libgd2-xpm-dev
    apt-get -y install libreadline-dev
    apt-get -y install libncurses5-dev 
    apt-get -y install libqwt-dev
    apt-get -y install libqwt5-qt4-dev
    apt-get -y install libode-dev
    apt-get -y install libcwiimote-dev
    apt-get -y install libbluetooth-dev 
    apt-get -y install qt3-dev-tools
    apt-get -y install make
    apt-get -y install cmake
    apt-get -y install liblapack-dev
    apt-get -y install libcv-dev
    apt-get -y install libcvaux-dev
    apt-get -y install libhighgui-dev
    apt-get -y install mencoder
    apt-get -y install mplayer
    apt-get -y install vlc
    apt-get -y install zsh

    apt-get -y install libdb4.6
    apt-get -y install libdb4.6-dev 
    apt-get -y install openssl
    apt-get -y install expat 
    apt-get -y install zeroc-ice33
    apt-get -y install ice33-services icee-slice ice33-translators 
    apt-get -y install libzeroc-ice33 libicee-dev libslice33
    apt-get -y install libzeroc-ice33-dev
    apt-get -y install icee-slice icee-translators icegrid-gui

    apt-get -y install liblivemedia-dev livemedia-utils
    apt-get -y install libglut3-dev
    apt-get -y install libsvm-dev

    apt-get -y install libxml2-dev

    apt-get -y install libjpeg-progs
    apt-get -y install libjpeg62-dev

    apt-get -y install qt4-dev-tools
    apt-get -y install qt4-designer
    apt-get -y install qt4-qtconfig
    apt-get -y install qconf
    apt-get -y install libqt4-core
    apt-get -y install libqt4-gui
    apt-get -y install libqt4-sql-psql

    apt-get -y install libpqxx-dev
    #apt-get -y install libboost-all-dev

    apt-get -y install python-paramiko
    apt-get -y install python-vte

    apt-get -y install libglew-dev
    apt-get -y install libglewmx-dev

    apt-get -y install libserial-dev

    apt-get -y install gnuplot
    apt-get -y install libsparsehash-dev
    apt-get -y install libtorch-dev
    apt-get -y install libxtst-dev
    apt-get -y install libavformat-dev
    apt-get -y install libavutil-dev
    apt-get -y install libswscale-dev
    apt-get -y install libavdevice-dev

    apt-get -y install sysv-rc-conf  # to configure services
    apt-get -y install sysstat

    apt-get -y install ncftp
    apt-get -y install ntp

    apt-get -y install flex
    apt-get -y install bison

    apt-get -y install libdevil-dev

    apt-get -y install libtool
    apt-get -y install shtool

    apt-get -y install libomniorb4-dev
    apt-get -y install omniidl4

    #apt-get -y install sun-java6-jre sun-java6-plugin sun-java6-fonts

    apt-get -y install libblitz0-dev

    #apt-get -y install acroread

    apt-get -y install sqlite3
    apt-get -y install libsqlite3-dev

    apt-get -y install libxp6 # for matlab

    apt-get -y install libtbb2
    apt-get -y install libv4l-dev
    apt-get -y install python2.6-dev libgtk2.0-dev libgtkmm-2.4-dev libzeroc-ice33-dev cmake
    apt-get -y install libportaudio2
    apt-get -y install doxygen graphviz
    apt-get -y install gccxml libicu-dev libicu44 python-dev python2.7-dev
    apt-get -y install zip unzip
    apt-get -y install csh # needed by matlab
fi

# configure cups client
read -p "Do you want to configure printing (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    cat > /etc/cups/client.conf <<EOF
ServerName 192.168.0.203
Encryption IfRequested
EOF
    /etc/init.d/cups restart
fi

# replace line 61 in /usr/lib/qt3/include/qimage.h  
#  { return key < other.key || (key==other.key && lang < other.lang); }
read -p "Do you want to fix qimage.h 61 problem (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    /bin/cat /usr/include/qt3/qimage.h | \
	/usr/bin/perl -pe 's/.*/  { return key < other.key || (key==other.key && lang < other.lang); }/ if $. == 61' > ~/tmp.txt
    /bin/mv -f ~/tmp.txt /usr/include/qt3/qimage.h
fi

# extra codecs and such
read -p "Do you want to install ubuntu restricted extras (flash, codecs, etc). This may be slow (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    apt-get -y install ubuntu-restricted-extra
fi

# kinect libs
read -p "Do you want to install the Kinect libraries (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-add-repository ppa:arne-alamut/freenect
    apt-get update

    apt-get -y install libusb-1.0-0-dev
    apt-get -y install libfreenect
    apt-get -y install libfreenect-dev
fi

# upgrade system
read -p "Do you want to install all software updates (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-get -y dist-upgrade

    apt-get autoremove

    # temporary fix?
    mkdir -p /usr/local/share/emacs/site-lisp
    mkdir -p /usr/local/share/emacs/23.1/site-lisp
fi

# h.264 support for matlab
read -p "Do you want to install h.264 support for Matlab, which may break kdenlive and others (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    apt-get update

    apt-get -y remove gstreamer0.10-plugins-bad-multiverse mencoder libavcodec-extra-52 gstreamer0.10-ffmpeg \
	vlc-nox videotrans libavformat-extra-52 ffmpeg libavfilter0 qdvdauthor-common qdvdauthor smplayer \
	vlc mjpegtools libmjpegtools-1.9 vlc-plugin-pulse libquicktime1 mplayer-nogui libavdevice52 libxvidcore4 \
	gstreamer0.10-plugins-ugly twolame gstreamer0.10-plugins-ugly-multiverse gstreamer0.10-plugins-bad

    apt-get update

    apt-get -y purge libtwolame0 libmp4v2-0 libdvbpsi5 libx264-85 libmad0 libmpcdec3 libmpeg2-4 libfaac0 libfaad2

    apt-get -y install mplayer gstreamer0.10-ffmpeg vlc

    apt-get -y install libavcodec-dev libavformat-dev libavdevice52

    apt-get -y autoremove
fi

read -p "Do you want to install ROS (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    echo "deb http://packages.ros.org/ros/ubuntu natty main" > /etc/apt/sources.list.d/ros-latest.list
    wget http://packages.ros.org/ros.key -O - | apt-key add -
    apt-get update
    apt-get install ros-diamondback-desktop-full
    echo "Add this to your .bashrc to load ROS env vars: source /opt/ros/diamondback/setup.bash"
fi

# upgrade system
read -p "Do you want to force grub to text (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    echo "GRUB_GFXPAYLOAD_LINUX=text" >> /etc/default/grub
    update-grub
fi

######################################################################
# the following is for iLab machines only:
######################################################################
if [ ! -d /lab/packages/forall ]; then
    echo "Skipping additional packages which are available on iLab machines only"
else
    # setup tmpwatch:
    read -p "Do you want to setup tmpwatch (Y/n)? "
    if [ "X$REPLY" != "Xn" ]; then
	/bin/cp /lab/packages/forall/current/tmpwatch.cron.daily /etc/cron.daily/tmpwatch
	/bin/cp /lab/packages/forall/current/tmpwatch /usr/sbin/
    fi

    read -p "Do you want to install gcc-4.6 (Y/n)? "
    if [ "X$REPLY" != "Xn" ]; then
	apt-get -f install
	dpkg -i /lab/packages/forall/current/gcc46/*.deb
	apt-get -f install
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.5 45 --slave /usr/bin/g++ g++ /usr/bin/g++-4.5
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 46 --slave /usr/bin/g++ g++ /usr/bin/g++-4.6
    fi

    # commercial software:
    read -p "Do you want to install local / commercial packages (Y/n)? "
    if [ "X$REPLY" != "Xn" ]; then

	echo "Removeing opencv 2.1..."
	apt-get -f remove opencv
	apt-get -f remove libcv2.1

        # local packages
	dpkg -i /lab/packages/forall/current/*.deb
	apt-get -f install

	echo "Installing flann + eigen3 ..."
	cd /usr/local
	tar zxvf /lab/packages/forall/current/flanneigen.tgz
	cd /usr/local/include
	ln -s eigen3/Eigen Eigen

	echo "Installing yaml ..."
	cd /usr/local
	tar zxvf /lab/packages/forall/current/yaml-cpp-0.2.5-installed.tgz

	echo "Installing boost ..."
	cd
	apt-get -y remove `dpkg --list |grep boost|awk '{ print $2 }'|xargs`
	dpkg -i /lab/packages/forall/current/boost/*.deb
	apt-get -f install

	# to build lwpr:
	# configure, make, make install, then move the includes from /usr/local/include to /usr/include/lwpr, and
        #for x in `/bin/ls /usr/include/lwpr`; do
        #    $R "#include <$x>" "#include <lwpr/$x>" -- /usr/include/lwpr/*
        #    $R "#include \"$x\"" "#include <lwpr/$x>" -- /usr/include/lwpr/*
        #done
	# and finally mv /usr/local/lib/liblwpr* /usr/lib
        cd /
        echo "Installing liblwpr..."
        tar jxpf /lab/packages/forall/current/lwpr-1.2.3-installed.tbz

        echo "Installing libirobot-create..."
        tar jxpf /lab/packages/forall/current/libirobot-create-installed.tbz

        # commercial packages:
        cd /
        echo "Installing EyeLink-1000 software..."
        tar jxpf /lab/vault/software/EyeLink-1000/eyelink-1000-installed.tbz
        echo "Installing Epix CameraLink XClib software..."
        tar jxpf /lab/vault/software/epix/xclib-3.7-ubuntu11-x64-installed.tbz
	echo "Installing xv..."
	tar jxpf /lab/packages/forall/current/xv-3.10a-ilab-ubuntu10.04-amd64.tbz

        # cuda thrust library
	echo "Installing Cuda Thrust library..."
	cd /usr/local
	unzip -o /lab/packages/forall/current/thrust-v1.2.zip
	ln -s /usr/local/thrust /usr/include/thrust
    fi

    # for cuda:
    echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf
    echo /lab/local/matlabR2010b/bin/glnxa64 > /etc/ld.so.conf.d/matlab.conf
    ldconfig

    echo "You should REBOOT first before you install cuda, and run this from the console."
    echo "Do not say y here unless you have an nVidia graphics card that does support CUDA."
    echo "For the cuda toolkit, select /usr/local/cuda for installation."
    echo "For the cuda SDK, select /usr/local/cudasdk for installation."
    read -p "Do you want to install the nVidia drivers and CUDA (Y/n)? "
    if [ "X$REPLY" != "Xn" ]; then

	read -p "Do you want to install the driver for CUDA-compatible nVidia cards ONLY (y/N)? "
	if [ "X$REPLY" = "Xy" ]; then
	    apt-get purge nvidia*
	    cat >> /etc/modprobe.d/blacklist.conf <<EOF
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist nvidiafb
blacklist rivatv
EOF
	    /lab/packages/forall/current/devdriver_4.0_linux_64_270.41.19.run
	fi

	read -p "Do you want to install CUDA dev kit (ok even if you don't have a CUDA board) (Y/n)? "
	echo "Just accept the proposed install locations"
	if [ "X$REPLY" != "Xn" ]; then
	    /bin/rm -rf /home/local/cuda /home/local/cudasdk
	    /lab/packages/forall/current/cudatoolkit_3.1_linux_64_ubuntu9.10.run
	    /lab/packages/forall/current/gpucomputingsdk_3.1_linux.run -- --prefix=/usr/local/cudasdk --cudaprefix=/usr/local/cuda
	    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
	    ldconfig
	fi
    fi
fi
