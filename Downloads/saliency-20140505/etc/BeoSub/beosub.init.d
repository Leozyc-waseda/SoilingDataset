#!/bin/bash
#
# chkconfig: 2345 85 15
# description: configure and start firewire for beobots

# source function library
. /etc/rc.d/init.d/functions
export PATH=/home/beosub/saliency/bin:${PATH}

case "$1" in
  start)
	gprintf "Starting Beosub: "

	# load modules:
	modprobe ov511
	modprobe v4l1-compat
	modprobe ovcamchip

	# start the code:
	beosubgo.sh

	success "beosub startup"
	echo
	;;
  stop)
	gprintf "Shutting down beosub: "

	# stop our code:
	beosubstop.sh
	ipcclean.sh

        # unload modules:
	rmmod ovcamchip
	rmmod v4l1-compat
	rmmod ov511

	success "beosub shutdown"
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
	gprintf "Usage: beosub {start|stop|status|restart|reload}\n"
	exit 1
esac

exit 0

