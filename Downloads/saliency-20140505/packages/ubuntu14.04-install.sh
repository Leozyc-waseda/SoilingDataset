#!/bin/bash

# Installation guide for Ubuntu 14.04 SERVER amd64

# RUN THIS SCRIPT AS ROOT

# Start with Ubuntu SERVER (we will install the desktop packages later). This is due to some problems with network
# configuration, graphics cards, etc in the ubuntu desktop edition

# Before running this script follow the instructions here:

# http://ilab.usc.edu/toolkit/documentation-ubuntu.shtml

# and, during installation, you should:

# - setup partitions as:
#   / (primary) [typically 10-30GB]
#   /home (primary) [rest of your disk]

# These days, we use swap files as opposed to a swap partition, as this is more flexible. When the ubuntu installer
# tells you that you do not have a swap partition, continue anyway.

# If you are upgrading, make sure that you do not erase the contents of /home!

# - When asked for your name, use "Administrator Account", username "administrator". Do not use your real name since
#   your user account will be handled through NIS.

# - If you have a dual ethernet card, the network will not configure properly. Even with a single ethernet, by default
#   it will use DHCP which is not good. After installation is complete, reboot and login using the administrator
#   account. Then go to System -> Preferences -> Network Connections, and:

# - configure your network (IPv4):
#    - ask Laurent for your IP address, or run 'ypcat -k hosts | grep YOURHOST' on 
#      an ilab machine, where YOURHOST is the hostname of your box
#    - netmask is 255.255.255.0
#    - gateway is 192.168.0.100
#    - DNS servers: 128.125.7.23, 128.125.253.194, 128.125.253.143

# fix whoopsie log clutter:
cat >> /etc/rsyslog.d/00-whoopsie.conf <<EOF
# whoopsie generates a lot of messages of "online". These are worthless and can be filtered
if $programname == 'whoopsie' and $msg == ' online' then ~
EOF

# setup software sources:
read -p "Do you want to setup software sources (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    # to setup sources, add them to /etc/apt/sources.list
    /bin/mv /etc/apt/sources.list /etc/apt/sources.list.bak
    /bin/cat > /etc/apt/sources.list <<EOF
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ trusty main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ trusty-updates main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ trusty-backports main restricted universe multiverse
deb http://mirrors.usc.edu/pub/linux/distributions/ubuntu/ trusty-security main restricted universe multiverse
deb http://archive.canonical.com/ubuntu trusty partner
EOF
    apt-get update
fi

# install some essentials:
read -p "Do you want to install emacs, subversion, vim, sshd (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
  apt-get -y --force-yes install emacs
  mkdir -p /usr/local/share/emacs/23.4/site-lisp # avoid having emacs complain about it missing
  apt-get -y --force-yes install gtk2-engines-pixbuf # for annoying error msgs
  apt-get -y --force-yes install subversion
  apt-get -y --force-yes install vim
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
    service nis restart
    service autofs restart
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

# setup swap file
read -p "Do you want to setup a 12 GB swap file (if not using swap partition) (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    fallocate -l 12g /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo "/swapfile none            swap    sw              0       0" >> /etc/fstab
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

    # avoid problems with: RPC: AUTH_GSS upcall timed out.
    sed --in-place=.bak "s/^NEED_GSSD=$/NEED_GSSD=yes/" /etc/default/nfs-common
    service nfs-kernel-server restart
fi

# add a toor account?
read -p "Do you want to add a toor account (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    echo 'toor:x:0:0:Toor Account:/root:/bin/bash' >> /etc/passwd
    chmod u+w /etc/shadow
    echo 'toor:$6$hZrTpviz$mQaXcS4Lxv/Z/hNwaxkBP8wEYuzlCR3BMI6xQzEy7Zgq0JvQD0dQO2Ay2B8VdwAUWT9NZoYGX7nYYjzRnpb6l/:14818:0:99999:7:::' >> /etc/shadow
    chmod u-w /etc/shadow
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
    #apt-get -y --force-yes install ffmpeg   # gone??
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
    apt-get -y --force-yes install make
    apt-get -y --force-yes install cmake
    apt-get -y --force-yes install mencoder
    apt-get -y --force-yes install mplayer
    apt-get -y --force-yes install libpostproc-dev #install with libavcodec-dev
    apt-get -y --force-yes install libavcodec-dev  #install with libpostproc-dev
    apt-get -y --force-yes install vlc
    apt-get -y --force-yes install zsh

    apt-get -y --force-yes install libdb-dev 
    apt-get -y --force-yes install openssl
    apt-get -y --force-yes install expat 


    apt-get -y --force-yes install ice35-slice icegrid-gui
    apt-get -y --force-yes install ice35-services icee-slice ice35-translators 
    apt-get -y --force-yes install libicee-dev libslice35
    apt-get -y --force-yes install libzeroc-ice35-dev
    apt-get -y --force-yes install icee-slice icee-translators
    apt-get -y --force-yes install zeroc-ice35 zeroc-icee

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

    apt-get -y --force-yes install libboost-all-dev  # boost

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
    apt-get -y --force-yes install sysstat

    apt-get -y --force-yes install ncftp
    apt-get -y --force-yes install ntp

    apt-get -y --force-yes install flex
    apt-get -y --force-yes install bison

    apt-get -y --force-yes install libdevil-dev

    apt-get -y --force-yes install libtool
    apt-get -y --force-yes install shtool

    apt-get -y --force-yes install libomniorb4-dev
    apt-get -y --force-yes install omniidl

    #apt-get -y --force-yes install sun-java6-jre sun-java6-plugin sun-java6-fonts

    apt-get -y --force-yes install libblitz0-dev

    #apt-get -y --force-yes install acroread

    apt-get -y --force-yes install sqlite3
    apt-get -y --force-yes install libsqlite3-dev

    apt-get -y --force-yes install libxp6 # for matlab

    apt-get -y --force-yes install libtbb2
    apt-get -y --force-yes install libv4l-dev
    apt-get -y --force-yes install python-dev libgtk2.0-dev libgtkmm-2.4-dev
    apt-get -y --force-yes install libportaudio2
    apt-get -y --force-yes install doxygen graphviz
    apt-get -y --force-yes install gccxml libicu-dev python-dev
    apt-get -y --force-yes install zip unzip
    apt-get -y --force-yes install csh # needed by matlab

    apt-get -y --force-yes install libeigen3-dev

    apt-get -y --force-yes install libopencv-dev python-opencv
    apt-get -y --force-yes install libthrust-dev
    apt-get -y --force-yes install libyaml-cpp0.3 libyaml-cpp0.3-dev
    apt-get -y --force-yes install libpcl1-dev
    apt-get -y --force-yes install libflann-dev

    apt-get -y --force-yes install guvcview
    apt-get -y --force-yes install libyaml-perl
    apt-get -y --force-yes install synaptic
    apt-get -y --force-yes install psensor

    apt-get -y --force-yes install liburg0-dev

    apt-get -y --force-yes install ubuntu-desktop
    apt-get -y --force-yes install clang-3.5 lldb-3.5 libclang-3.5-dev clang-format-3.5
fi

# configure cups client
read -p "Do you want to configure printing (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    apt-get -y --force-yes install cups cups-filters printer-driver-gutenprint
    cat > /etc/cups/client.conf <<EOF
ServerName 192.168.0.203
Encryption IfRequested
EOF
    service cups restart
fi

# kinect libs
read -p "Do you want to install the Kinect libraries (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-get -y --force-yes install libusb-1.0-0-dev
    apt-get -y --force-yes install libfreenect0.1
    apt-get -y --force-yes install libfreenect-dev
    sed -i '130s/make_pair/pair/' /usr/include/libfreenect.hpp
fi

# upgrade system
read -p "Do you want to install all software updates (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    apt-get -y dist-upgrade

    apt-get autoremove
fi

# force text grub
read -p "Do you want to force grub to text (y/N)? "
if [ "X$REPLY" = "Xy" ]; then
    echo "GRUB_GFXPAYLOAD_LINUX=text" >> /etc/default/grub
    update-grub
fi

# fix resolvconf issues
read -p "Do you want to fix resolvconf (Y/n)? "
if [ "X$REPLY" != "Xn" ]; then
    dpkg-reconfigure resolvconf
fi

######################################################################
# the following is for iLab machines only:
######################################################################
if [ ! -d /lab/packages/forall ]; then
    echo "Skipping additional packages which are available on iLab machines only"
else
    # setup tmpwatch:
    #read -p "Do you want to setup tmpwatch (Y/n)? "
    #if [ "X$REPLY" != "Xn" ]; then
	#/bin/cp /lab/packages/forall/current/tmpwatch.cron.daily /etc/cron.daily/tmpwatch
	#/bin/cp /lab/packages/forall/current/tmpwatch /usr/sbin/
    #fi

    # commercial software:
    read -p "Do you want to install local / commercial packages (Y/n)? "
    if [ "X$REPLY" != "Xn" ]; then

        # local packages
	dpkg -i /lab/packages/forall/current/*.deb
	apt-get remove ".*ice34.*"
	apt-get remove "libice.*34.*"
	dpkg -i /lab/packages/forall/current/ice-3.5/*.deb
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
	apt-get -y --force-yes install libtiff4 libjpeg62

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

    # for matlab: note how we name it zzzmatlab so other configs will take precedence:
    for mat in \
	/lab/local/matlabR2010b/bin/glnxa64 \
	/lab/local/matlabR2011b/bin/glnxa64 \
	/lab/local/matlabR2012b/bin/glnxa64 \
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
