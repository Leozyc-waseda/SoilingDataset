use IO::Socket; 

$BASE_OUT = 50000;
$BASE_IN  = 50500;

my $sock = new IO::Socket::INET ( 
    PeerAddr => 'ibeo.java.usc.edu', 
    PeerPort => $BASE_OUT, 
    Proto => 'tcp',
    ); 
die "Could not create socket: $!\n" unless $sock; 

my $isock = new IO::Socket::INET ( 
    LocalHost => '', 
    LocalPort => $BASE_IN, 
    Proto => 'tcp', 
    Listen => 1, 
    Reuse => 1, 
    );
die "Could not create socket: $!\n" unless $isock;

@hname      = qx'echo $HOSTNAME';
@shn        = split(/\n/,$hname[0]);
$HOST       = $shn[0];
$packed_ip  = gethostbyname($HOST);
if (defined $packed_ip) { $IP = inet_ntoa($packed_ip); }
$ID     = "1";
$MTYPE  = "TEST";
$MCONT  = "This is a test";
$RET    = 1;
   
print $sock "$HOST,$IP,$ID,$MTYPE,$MCONT,$RET";
close($sock);

my $new_sock = $isock->accept(); 
while(<$new_sock>) 
{ 
    print("RETURN $_\n");
} 
close($isock);

######################################################################

my $sock = new IO::Socket::INET ( 
    PeerAddr => 'ibeo.java.usc.edu', 
    PeerPort => $BASE_OUT, 
    Proto => 'tcp',
    ); 
die "Could not create socket: $!\n" unless $sock;

my $isock = new IO::Socket::INET ( 
    LocalHost => '', 
    LocalPort => $BASE_IN, 
    Proto => 'tcp', 
    Listen => 1, 
    Reuse => 1, 
    );
die "Could not create socket: $!\n" unless $isock;

@hname      = qx'echo $HOSTNAME';
@shn        = split(/\n/,$hname[0]);
$HOST       = $shn[0];
$packed_ip  = gethostbyname($HOST);
if (defined $packed_ip) { $IP = inet_ntoa($packed_ip); }
$ID     = "1";
$MTYPE  = "COMMAND";
$MCONT  = "ls -1 /lab/mundhenk/";
$RET    = 1;
   
print $sock "$HOST,$IP,$ID,$MTYPE,$MCONT,$RET";
close($sock);

my $new_sock = $isock->accept(); 
while(<$new_sock>) 
{ 
    print("RETURN $_\n");
} 
close($isock);

######################################################################
$BASE_OUT++;
$BASE_IN++;

my $sock = new IO::Socket::INET ( 
    PeerAddr => 'ibeo.java.usc.edu', 
    PeerPort => $BASE_OUT, 
    Proto => 'tcp',
    ); 
die "Could not create socket: $!\n" unless $sock;

my $isock = new IO::Socket::INET ( 
    LocalHost => '', 
    LocalPort => $BASE_IN, 
    Proto => 'tcp', 
    Listen => 1, 
    Reuse => 1, 
    );
die "Could not create socket: $!\n" unless $isock;

@hname      = qx'echo $HOSTNAME';
@shn        = split(/\n/,$hname[0]);
$HOST       = $shn[0];
$packed_ip  = gethostbyname($HOST);
if (defined $packed_ip) { $IP = inet_ntoa($packed_ip); }
$ID     = "1";
$MTYPE  = "SOCK_EXIT";
$MCONT  = "0";
$RET    = 1;
   
print $sock "$HOST,$IP,$ID,$MTYPE,$MCONT,$RET";
close($sock);

my $new_sock = $isock->accept(); 
while(<$new_sock>) 
{ 
    print("RETURN $_\n");
} 
close($isock);
