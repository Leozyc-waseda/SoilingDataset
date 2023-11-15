use IO::Socket; 
use Carp;
use POSIX;
use strict; 
$SIG{CHLD} = \&reaper; 
$SIG{USR1} = \&error; 
$SIG{USR2} = \&warning;
BEGIN { $ENV{PATH} = '/sbin:/usr/sbin:/bin:/usr/bin' }

# get a mutex going:
my $muname = "/tmp/beoqueue-host.mutex.$$";     # mutex (may be unnecessary)
open MUTEX, ">$muname" || die "FATAL>>> Cannot write mutex file $muname: $!\n";

my $PROTO         = getprotobyname('tcp');
my $PORT_RANGE    = 500;
my $LOW_IN_PORT   = 50000;
my $HIGH_IN_PORT  = $LOW_IN_PORT + $PORT_RANGE;
my $LOW_OUT_PORT  = 50500;
my $HIGH_OUT_PORT = $LOW_OUT_PORT + $PORT_RANGE;
my $do_exit       = 0;
my $waitedpid     = 0;
my $procid        = POSIX::getpid();
my @plist         = {};
my @port_use      = {};
my $delay         = 1;

# set all sockets as free
my @sock_use      = {};
my $sock_count    = 0;
for(my $n = $LOW_IN_PORT; $n < $HIGH_IN_PORT; $n++)
{
    $sock_use[$sock_count] = 0;
}

(my $MY_IP,my $MY_HOST) = get_my_info();

print("Starting listener on $MY_HOST : $MY_IP\n");

while(!($do_exit))
{
    my $count = 0;
    for(my $n = $LOW_IN_PORT; $n < $HIGH_IN_PORT; $n++)
    {
	if(!($sock_use[$count])) # is this socket in use?
	{
	    my $res = fork();
	    if (not defined $res) 
	    { 
		die "FATAL>>> Cannot fork: $!\n"; 
	    }
	    elsif ($res == 0) # we are the child
	    {  
		my $IN_PORT  = $n;
		my $OUT_PORT = $LOW_OUT_PORT + $count;
		msg("Forking listener on port $IN_PORT");
		$do_exit = do_listen($IN_PORT,$OUT_PORT);
		exit 0;
	    }
	    else
	    {
		# print("Setting use $count\n");
		flock MUTEX, 2;        # get protected access to plist	
		$plist[$count]    = $res;
		$sock_use[$count] = 1;
		$port_use[$count] = $n;	
		flock MUTEX, 8;        # free plist
	    }
	}
	$count++;
    }
    sleep $delay;
}

######################################################################
sub do_listen
{
    my $port_do_exit = 0;
    my $IN_PORT  = $_[0];
    my $OUT_PORT = $_[1]; 

    my $sock = new IO::Socket::INET ( 
	LocalHost => '', 
	LocalPort => $IN_PORT, 
	Proto =>     $PROTO, 
	Listen =>    1, 
	Reuse =>     1, 
	);
    die "Could not create socket: $!\n" unless $sock;
    
    # print("Listening on port $IN_PORT\n");

    while(!($port_do_exit))
    {	
	my $new_sock = $sock->accept(); 
	my $msg_size = 0;
	# print("Got message on port $IN_PORT from $OUT_PORT\n");
	
	while(<$new_sock>) 
	{ 
	    $port_do_exit = parse_message($_,$OUT_PORT,$IN_PORT);
	    $msg_size++;
	}
	if(!($msg_size))
	{ 
	    msg("Received zero length message on port $IN_PORT");
	}
	# print("message size $msg_size\n");	
    }
    close($sock);
    return $port_do_exit;
} 

######################################################################
sub do_speak
{
    my $HOST     = $_[0];
    my $OUT_PORT = $_[1];
    my $ID       = $_[2];
    my $RESULT   = $_[3];
    my $RET_MSG  = $_[4];

    my $sock = new IO::Socket::INET ( 
	PeerAddr => $HOST, 
	PeerPort => $OUT_PORT, 
	Proto =>    $PROTO,
	);   
    die "Could not create socket: $!\n" unless $sock;

    print $sock "$MY_HOST,$ID,$RESULT,$RET_MSG";
    close($sock);   
}

######################################################################
sub get_my_info()
{   
    my @hname      = qx'echo $HOSTNAME';
    my @shn        = split(/\n/,$hname[0]);
    $MY_HOST    = $shn[0];
    my $packed_ip  = gethostbyname($MY_HOST);
    if (defined $packed_ip) { $MY_IP = inet_ntoa($packed_ip); }
    else { $MY_IP = "unknown"; } 
    return ($MY_IP,$MY_HOST);
}

######################################################################
sub parse_message
{
    
    # get contents of message
    my @msg      = split(/,/,$_[0]); 
    my $OUT_PORT = $_[1];
    my $IN_PORT  = $_[2];
 
    # print("Parsing Message on port $IN_PORT\n");

    my $HOST  = $msg[0];
    my $IP    = $msg[1];
    my $ID    = $msg[2];
    my $MTYPE = $msg[3];
    my $MCONT = $msg[4];
    my $RET   = $msg[5];

    my $do_exit = 0;    
    my $RET_MSG = "err";
    my $RESULT  = "";
    my @RESQX   = {};
	
    # What type of message did we just get?
    if($MTYPE eq 'TEST')
    {
	print("\nTest Message from $HOST\n");
	print("$IP : $ID : $MTYPE\n");
	print("$MCONT\n");
	print("Return $RET\n\n");
	$RET_MSG = "ok";
	$RESULT  = "test";
    }
    elsif($MTYPE eq 'COMMAND')
    {
	#print("COMMAND $MCONT\n"); 
	@RESQX  = qx($MCONT);
	$RESULT = "[RESULT";
	foreach my $r (@RESQX)
	{
	    chomp($r);
	    $RESULT = "$RESULT,$r";
	}
	$RET_MSG = "ok";
	$RESULT  = "$RESULT]";
    }
    elsif($MTYPE eq 'SOCK_EXIT')
    {
	msg("Recieved socket exit request from $HOST"); 
	$RESULT  = "sock_exit";
	$RET_MSG = "ok";
	$do_exit = 1;
    } 
    elsif($MTYPE eq 'EXIT') # not yet supported
    {
	msg("Recieved exit request from $HOST"); 
	$RESULT  = "exit";
	$RET_MSG = "ok";
	$do_exit = 2;
    }
    else
    {
	msg("Recieved invalid request \"$MTYPE\" from $HOST"); 
    }

    # Send a return message if requested
    if($RET == 1)
    {
	do_speak($HOST,$OUT_PORT,$ID,$RESULT,$RET_MSG);
    }

    return $do_exit;
}

######################################################################    
sub reaper
{
    my $pid;
    while(($pid = waitpid(-1, &WNOHANG)) > 0)
    {
	if(WIFEXITED($?)) 
	{
	    if(WSTOPSIG($?) == 0) 
	    { 
		my $jj = 0;
		while($jj <= $#plist) 
		{
		    if ($plist[$jj] == $pid) 
		    {
			msg("Socket Exit on pid $pid port $port_use[$jj]");
			#print("Unsetting use $jj\n");
			flock MUTEX, 2;  # get protected access to plist
			$plist[$jj]    = 0;
			$sock_use[$jj] = 0;
			$port_use[$jj] = 0;
			flock MUTEX, 8;  # free plist
		    }
		    $jj ++;
		}
	    }
	    #print("done\n");
	}
    }
    $SIG{CHLD} = \&reaper;  # loathe sysV
} 

######################################################################
sub error {
    msg("FATAL ERROR recieved from child node");
    die "STOPPED!\n";
}

######################################################################
sub warning {
    msg("WARNING received from child node");
}  

######################################################################
sub msg { # string
    chomp();
    my $dat = `/bin/date +"%y%m%d-%H:%M:%S"`; chomp($dat);
    print STDERR "BEOQUEUE HOST $dat - $_[0]\n";
}
