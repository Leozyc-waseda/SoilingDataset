#!/bin/bash

# Installation guide for Ubuntu 12.04lts amd64

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
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ quantal main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ quantal-updates main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ quantal-backports main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ quantal-security main restricted universe multiverse
deb http://archive.canonical.com/ubuntu quantal partner
EOF
    apt-get update
fi

# install some essentials:
read -p "Do you want to install emacs, subversion, sshd (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
  apt-get -y --force-yes install emacs
  apt-get -y --force-yes install gtk2-engines-pixbuf # for annoying error msgs
  apt-get -y --force-yes install subversion
  apt-get -y --force-yes install openssh-server
fi

# setup nis client:
read -p "Do you want setup the NIS client (enter 'lab' when asked for domain name) (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    echo "Setup NIS server"
    echo "Editing.. /etc/defaultdomain"
    echo "lab" > /etc/defaultdomain
    apt-get -y --force-yes install nis
    apt-get -y --force-yes install autofs
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
    apt-get -y --force-yes install nfs-kernel-server
    for f in /home*/*; do
	bf=`basename $f`
	if [ -d $f -a "x$bf" != "xlost+found" ]; then
	    echo "$f 192.168.0.0/24(rw,async,no_subtree_check) 127.0.0.0/16(rw,async,no_subtree_check)" >> /etc/exports
	fi
    done
    exportfs -r
fi

# add a toor account?
read -p "Do you want to add a toor account (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    echo 'toor:x:0:0:Toor Account:/root:/bin/bash' >> /etc/passwd
    chmod u+w /etc/shadow
    echo 'toor:$6$hZrTpviz$mQaXcS4Lxv/Z/hNwaxkBP8wEYuzlCR3BMI6xQzEy7Zgq0JvQD0dQO2Ay2B8VdwAUWT9NZoYGX7nYYjzRnpb6l/:14818:0:99999:7:::' >> /etc/shadow
    chmod u-w /etc/shadow
fi

# turn off dm:
read -p "Finally, do you want to turn off gdm (will take effect after reboot) (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    sed --in-place=bak s/016/0126/ /etc/init/gdm.conf
    update-rc.d -f gdm remove
fi

# install packages needed to compile ilab software:
read -p "Do you want to install Linux tools and libraries needed to compile iLab code (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    #apt-get update

    apt-get -y --force-yes install build-essential
    apt-get -y --force-yes install autoconf
    apt-get -y --force-yes install autoconf-archive
    apt-get -y --force-yes install automake

    apt-get -y --force-yes install vim-gtk
    apt-get -y --force-yes install tcl-dev
    apt-get -y --force-yes install tk-dev
    apt-get -y --force-yes install tcl8.4-dev
    apt-get -y --force-yes install tk8.4-dev
    apt-get -y --force-yes install libpopt-dev
    apt-get -y --force-yes install bzip2 libbz2-dev
    apt-get -y --force-yes install ffmpeg
    apt-get -y --force-yes install fftw3-dev
    apt-get -y --force-yes install gsl-bin
    apt-get -y --force-yes install gdb
    apt-get -y --force-yes install libgsl0-dev
    apt-get -y --force-yes install libblas-dev
    apt-get -y --force-yes install liblapack-dev
    apt-get -y --force-yes install libavc1394-dev

    apt-get -y --force-yes install libdc1394-22-dev
    apt-get -y --force-yes install libraw1394-dev
    apt-get -y --force-yes install libsdl1.2-dev
    apt-get -y --force-yes install libsdl-gfx1.2-dev
    apt-get -y --force-yes install libsdl-mixer1.2-dev
    apt-get -y --force-yes install libsdl-image1.2-dev
    apt-get -y --force-yes install libsdl-ttf2.0-dev
    apt-get -y --force-yes install libopenmpi-dev
    apt-get -y --force-yes install libgd-tools
    apt-get -y --force-yes install libgd2-xpm-dev
    apt-get -y --force-yes install libreadline-dev
    apt-get -y --force-yes install libncurses5-dev 
    apt-get -y --force-yes install libqwt-dev
    apt-get -y --force-yes install libode-dev
    apt-get -y --force-yes install libcwiimote-dev
    apt-get -y --force-yes install libbluetooth-dev 
    apt-get -y --force-yes install qt3-dev-tools qt3-designer libqt3-headers libqt3-mt-dev
    apt-get -y --force-yes install make
    apt-get -y --force-yes install cmake
    apt-get -y --force-yes install mencoder
    apt-get -y --force-yes install mplayer
    apt-get -y --force-yes install libpostproc-dev #install with libavcodec-dev
    apt-get -y --force-yes install libavcodec-dev  #install with libpostproc-dev
    apt-get -y --force-yes install vlc
    apt-get -y --force-yes install zsh

    apt-get -y --force-yes install libdb4.8
    apt-get -y --force-yes install libdb4.8-dev 
    apt-get -y --force-yes install openssl
    apt-get -y --force-yes install expat 
    apt-get -y --force-yes install zeroc-ice34
    apt-get -y --force-yes install ice34-services icee-slice ice34-translators 
    apt-get -y --force-yes install libzeroc-ice34 libicee-dev libslice34
    apt-get -y --force-yes install libzeroc-ice34-dev
    apt-get -y --force-yes install icee-slice icee-translators icegrid-gui

    apt-get -y --force-yes install liblivemedia-dev livemedia-utils
    apt-get -y --force-yes install freeglut3-dev
    apt-get -y --force-yes install libsvm-dev

    apt-get -y --force-yes install libxml2-dev

    apt-get -y --force-yes install libjpeg-progs
    apt-get -y --force-yes install libjpeg62-dev

    apt-get -y --force-yes install qt4-dev-tools
    apt-get -y --force-yes install qt4-designer
    apt-get -y --force-yes install qt4-qtconfig
    apt-get -y --force-yes install qconf
    apt-get -y --force-yes install libqt4-core
    apt-get -y --force-yes install libqt4-gui
    apt-get -y --force-yes install libqt4-sql-psql

    apt-get -y --force-yes install libpqxx3-dev

    apt-get -y --force-yes install libboost1.48-all-dev  # boost

    apt-get -y --force-yes install python-paramiko
    apt-get -y --force-yes install python-vte

    apt-get -y --force-yes install libglew-dev
    apt-get -y --force-yes install libglewmx-dev

    apt-get -y --force-yes install libserial-dev

    apt-get -y --force-yes install gnuplot
    apt-get -y --force-yes install libsparsehash-dev
    apt-get -y --force-yes install libtorch-dev
    apt-get -y --force-yes install libxtst-dev
    apt-get -y --force-yes install libavformat-dev
    apt-get -y --force-yes install libavutil-dev
    apt-get -y --force-yes install libswscale-dev
    apt-get -y --force-yes install libavdevice-dev
    apt-get -y --force-yes install libavfilter-dev

    apt-get -y --force-yes install sysv-rc-conf  # to configure services
    apt-get -y --force-yes install chkconfig  # to configure services
    apt-get -y --force-yes install sysstat

    apt-get -y --force-yes install ncftp
    apt-get -y --force-yes install ntp

    apt-get -y --force-yes install flex
    apt-get -y --force-yes install bison

    apt-get -y --force-yes install libdevil-dev

    apt-get -y --force-yes install libtool
    apt-get -y --force-yes install shtool

    apt-get -y --force-yes install libomniorb4-dev
    apt-get -y --force-yes install omniidl4

    #apt-get -y --force-yes install sun-java6-jre sun-java6-plugin sun-java6-fonts

    apt-get -y --force-yes install libblitz0-dev

    #apt-get -y --force-yes install acroread

    apt-get -y --force-yes install sqlite3
    apt-get -y --force-yes install libsqlite3-dev

    apt-get -y --force-yes install libxp6 # for matlab

    apt-get -y --force-yes install libtbb2
    apt-get -y --force-yes install libv4l-dev
    apt-get -y --force-yes install python-dev libgtk2.0-dev libgtkmm-2.4-dev libzeroc-ice34-dev cmake
    apt-get -y --force-yes install libportaudio2
    apt-get -y --force-yes install doxygen graphviz
    apt-get -y --force-yes install gccxml libicu-dev python-dev python2.7-dev
    apt-get -y --force-yes install zip unzip
    apt-get -y --force-yes install csh # needed by matlab

    apt-get -y --force-yes install libeigen3-dev

/bin/ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

    apt-get -y --force-yes install libopencv-dev python-opencv
    apt-get -y --force-yes install libthrust-dev
    ##apt-get -y --force-yes install libyaml-dev # too old, need a newer one
    apt-get -y --force-yes install libflann-dev

    apt-get -y --force-yes install guvcview
    apt-get -y --force-yes install libyaml-perl

fi

# install latest gcc?
read -p "Do you want to install gcc 4.7 (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get update
    apt-get -y --force-yes install gcc-4.7 g++-4.7
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 46 --slave /usr/bin/g++ g++ /usr/bin/g++-4.6
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 47 --slave /usr/bin/g++ g++ /usr/bin/g++-4.7
fi

# configure cups client
read -p "Do you want to configure printing (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    cat > /etc/cups/client.conf <<EOF
ServerName 192.168.0.203
Encryption IfRequested
EOF
    /etc/init.d/cups restart
fi

# replace line 61 in /usr/lib/qt3/include/qimage.h  
#  { return key < other.key || (key==other.key && lang < other.lang); }
read -p "Do you want to fix qimage.h 61 problem (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    /bin/cat /usr/include/qt3/qimage.h | \
	/usr/bin/perl -pe 's/.*/  { return key < other.key || (key==other.key && lang < other.lang); }/ if $. == 61' > ~/tmp.txt
    /bin/mv -f ~/tmp.txt /usr/include/qt3/qimage.h
fi

# kinect libs
read -p "Do you want to install the Kinect libraries (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-add-repository ppa:arne-alamut/freenect
    apt-get update

    apt-get -y --force-yes install libusb-1.0-0-dev
    apt-get -y --force-yes install libfreenect
    apt-get -y --force-yes install libfreenect-dev
    sed -i '130s/make_pair/pair/' /usr/include/libfreenect.hpp
fi

# upgrade system
read -p "Do you want to install all software updates (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-get -y dist-upgrade

    apt-get autoremove
fi

# ros
#read -p "Do you want to install ROS (y/N)? "
#if [ "X$REPLY" = "Xy" ]; then
#    echo "deb http://packages.ros.org/ros/ubuntu quantal main" > /etc/apt/sources.list.d/ros-latest.list
#    wget http://packages.ros.org/ros.key -O - | apt-key add -
#    apt-get update
#    apt-get install ros-diamondback-desktop-full
#    echo "Add this to your .bashrc to load ROS env vars: source /opt/ros/diamondback/setup.bash"
#fi

# force text grub
read -p "Do you want to force grub to text (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    echo "GRUB_GFXPAYLOAD_LINUX=text" >> /etc/default/grub
    update-grub
fi

# add nrt dependencies
read -p "Do you want to install NRT dependencies (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    add-apt-repository "deb http://nrtkit.org/apt quantal main"
    apt-get update
    apt-get install nrt
    apt-get remove nrt # we remove nrt so we can use the source version
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

    # commercial software:
    read -p "Do you want to install local / commercial packages (Y/n)? "
    if [ "X$REPLY" != "Xn" ]; then

        # local packages
	dpkg -i /lab/packages/forall/current/*.deb
	apt-get -f install

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

        # OpenNI (unstable for xtion support):
	cd /tmp
	tar jxvf /lab/packages/forall/current/openni-bin-dev-linux-x64-v1.5.4.0.tar.bz2
	cd OpenNI-Bin-Dev-Linux-x64-v1.5.4.0
	./install.sh
	cd ..
	/bin/rm -rf OpenNI-Bin-Dev-Linux-x64-v1.5.4.0

	tar jxvf /lab/packages/forall/current/sensor-bin-linux-x64-v5.1.2.1.tar.bz2
	cd Sensor-Bin-Linux-x64-v5.1.2.1
	./install.sh
	cd ..
	/bin/rm -rf Sensor-Bin-Linux-x64-v5.1.2.1

	tar jxvf /lab/packages/forall/current/nite-bin-linux-x64-v1.5.2.21.tar.bz2
	cd NITE-Bin-Dev-Linux-x64-v1.5.2.21
	./install.sh
	cd ..
	/bin/rm -rf NITE-Bin-Dev-Linux-x64-v1.5.2.21

	# Patch openni line 65 for c++-0x:
	#sed -i '65s/linux/__linux__/' /usr/include/ni/XnPlatform.h
	#sed -i '65s/i386/__i386__/' /usr/include/ni/XnPlatform.h

        # bumblebee
	cd /usr
	tar jxvf /lab/packages/forall/current/bumblebee.tbz
    fi

    # for cuda:
    echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf
    echo /usr/local/cuda/lib >> /etc/ld.so.conf.d/cuda.conf

    # for matlab: note how we name it zzzmatlab so other configs will take precedence:
    for mat in \
	/lab/local/matlabR2010b/bin/glnxa64 \
	/lab/local/matlabR2011b/bin/glnxa64 \
	/usr/local/matlabR2011b/bin/glnxa64 ; do
	if  [ -d $mat ]; then echo $mat > /etc/ld.so.conf.d/zzzmatlab.conf; fi
    done

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
	    /lab/packages/forall/current/devdriver_4.2_linux_64_295.41.run
	fi

	read -p "Do you want to install CUDA dev kit (ok even if you don't have a CUDA board) (Y/n)? "
	echo "Just accept the proposed install locations"
	if [ "X$REPLY" != "Xn" ]; then
	    /bin/rm -rf /home/local/cuda /home/local/cudasdk
	    /lab/packages/forall/current/cudatoolkit_4.2.9_linux_64_ubuntu11.04.run
	    /lab/packages/forall/current/gpucomputingsdk_4.2.9_linux.run -- \
		--prefix=/usr/local/cudasdk --cudaprefix=/usr/local/cuda

	    cd /usr/local/cuda
	    mv lib lib32
	    ln -s lib64 lib # bugfix for matlab cuda
	    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
	    echo "/usr/local/cuda/lib" >> /etc/ld.so.conf.d/cuda.conf
	    ldconfig

	    cd /usr/local/cudasdk/C
	    make -j 6 -k
	fi
    fi
fi
