#!/bin/bash
#
# chkconfig: 2345 85 15
# description: configure and start firewire for beobots

# source function library
#. /etc/rc.d/init.d/functions

case "$1" in
  start)
	echo "Starting Beobot firewire: "
	insmod ieee1394 > /dev/null 2>&1
	insmod ohci1394 attempt_root=1 > /dev/null 2>&1
	modprobe raw1394
	modprobe video1394
	if [ ! -c /dev/raw1394 ]; then
	  mknod /dev/raw1394 c 171 0
	fi
	if [ ! -c /dev/video1394/0 ]; then
	  rm -rf /dev/video1394
	  mkdir /dev/video1394
	  mknod /dev/video1394/0 c 171 16
	fi
	chmod 777 /dev/raw1394 /dev/video1394/0
	echo "firewire startup"
	echo
	;;
  stop)
	echo "Shutting down Beobot firewire: "
	rmmod eth1394
	rmmod video1394
	rmmod raw1394
	rmmod ohci1394
	rmmod ieee1394
	echo "firewire shutdown"
	echo
	;;
  restart|reload)
	$0 stop
	$0 start
	;;
  status)
	lsmod
	;;
  *)
	echo "Usage: firewire {start|stop|status|restart|reload}"
	exit 1
esac

exit 0

