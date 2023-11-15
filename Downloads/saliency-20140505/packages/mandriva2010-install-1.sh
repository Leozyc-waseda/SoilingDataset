#!/bin/bash

# Run this script AS ROOT to configure a freshly installed Mandriva O.S. for iLab operation

# during Mandriva 2010 installation, you should:

# - setup partitions as:
#   / (primary) [typically 10-20GB]
#   swap (primary) [typically 1-2x your RAM size)
#   /home (primary) [remainder of your disk]

# - select 'custom install', then 'manual package selection', then click on the little floppy icon, and load
#   mandriva2010-x64-packages.pl (which you may have copied to a USB drive or such; note that if you are using a USB
#   drive, it will not be detected the first time, just click cancel when you see the list of drives to load from, and
#   click the floppy icon again for a second try; this time your USB drive should show up in the list of drives to load
#   from)

# - create a local 'admin' account for a regular user

# - configure your timezone and set the machine to use ntp for time synchronization

# - configure your graphics card

# - configure your network:
#    - ask Laurent for your IP address, or run 'ypcat -k hosts | grep YOURHOST' on an ilab machine, where YOURHOST is
#      the hostname of your box
#    - netwmask is 255.255.255.0
#    - gateway is 192.168.0.100
#    - DNS servers: 128.125.7.23, 128.125.253.194, 128.125.253.143

# - turn off the shorewall firewall if you only have one network connection

# setup nis client:
read -p "Do you want setup NIS client (Y/n)?"
if [ "$REPLY" == "y" ]; then
    echo "Setup NIS server"
    urpmi --force yp-tools
    echo "Editing.. /etc/yp.conf"
    echo "domain lab server 192.168.0.200" >> /etc/yp.conf
    echo "Editing.. sysconfig/network"
    echo "NISDOMAIN=lab" >> /etc/sysconfig/network
    echo "Editing.. /etc/nsswitch.conf"
    mv /etc/nsswitch.conf /etc/nsswitch.conf.org
    cat > /etc/nsswitch.conf <<EOF
passwd:         files nis compat
shadow:         files nis
group:          files nis compat
hosts:          files nis dns
networks:       files nis
services:       files nis
protocols:      files nis
rpc:            files nis
ethers:         files nis
netmasks:       files nis
netgroup:       files nis
publickey:      files nis
bootparams:     files nis
automount:      files nis
aliases:        files nis
EOF
    nisdomainname lab
    service ypbind restart
    service autofs restart
    chkconfig ypbind on
    chkconfig autofs on
    # Rand's fix for auth when using NIS (to allow NIS users to login on the console):
    sed -i 's/shadow/shadow passwd/g' /etc/pam.d/system-auth
fi

# setup tmpwatch:
read -p "Do you want to setup tmpwatch (Y/n)?"
if [ "$REPLY" == "y" ]; then
    for f in /home*; do for d in 1 7 30 u; do mkdir -p $f/tmp/$d; chmod 777 $f/tmp/$d; done; done
    for f in /home*/tmp/1; do echo "/usr/sbin/tmpwatch -m 24 $f" >> /etc/cron.daily/tmpwatch; done
    for f in /home*/tmp/7; do echo "/usr/sbin/tmpwatch -m 168 $f" >> /etc/cron.daily/tmpwatch; done
    for f in /home*/tmp/30; do echo "/usr/sbin/tmpwatch -m 720 $f" >> /etc/cron.daily/tmpwatch; done
fi

# setup exports:
read -p "Do you want to setup exports (Y/n)?"
if [ "$REPLY" == "y" ]; then
    for f in /home*/*; do
	bf=`basename $f`
	if [ -d $f -a "x$bf" != "xlost+found" ]; then
	    echo "$f 192.168.0.0/24(rw,async,no_subtree_check)" >> /etc/exports
	fi
    done
fi

# turn off postfix and friends:
read -p "Do you want to turn off postfix, partmon, smb, mandi (Y/n)?"
if [ "$REPLY" == "y" ]; then
    for f in postfix partmon smb mandi coherence cups msec vnstat; do
	chkconfig $f off
	service $f stop
    done
fi

# add a toor account?
read -p "Do you want to add a toor account?"
if [ "$REPLY" == "y" ]; then
    echo 'toor:x:0:0:Toor Account:/root:/bin/bash' >> /etc/passwd
    chmod u+w /etc/shadow
    echo 'toor:$2a$08$IVour9XPry8UQchDARsIhub4AyZ0/49XsU.F/wKkB7VhEQGj6EzW2:14679:0:99999:7:::' >> /etc/shadow
    chmod u-w /etc/shadow
fi

# move /usr/local to /home/local:
read -p "Do you want to delete /usr/local and make it point to /home/local?"
if [ "$REPLY" == "y" ]; then
    if [ -d /home/local ]; then # already exists
	/bin/rm -rf /usr/local
	ln -s /home/local /usr/local
    else
	/bin/mv /usr/local /home/
	ln -s /home/local /usr/local
    fi
fi

# uninstall a bunch of junk:
read -p "Do you want to uninstall codeina (Y/n)?"
if [ "$REPLY" == "y" ]; then
    rpm -e codeina
    rpm -e lib64beagle1 yelp
fi

# turn off dm, warning this is terminal, kills X:
read -p "Finally, do you want to turn off dm -- WARNING: THIS WILL KILL X (Y/n)?"
if [ "$REPLY" == "y" ]; then
    chkconfig dm off
    service dm stop
fi
