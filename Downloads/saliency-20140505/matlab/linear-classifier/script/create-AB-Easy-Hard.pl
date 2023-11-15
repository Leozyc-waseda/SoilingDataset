use strict;
# How many lead in fixation frames?
my $LEAD_IN_FRAMES = 10;
# What is the base dir for all sequences
my $BASE_DIR       = "/lab/tmpib/u/newSequences/Partial-Repete-AB-Var-T1";
# Where to place the new sequences
my $OUT_DIR        = "/lab/tmpib/u/newSequences/Partial-Repete-AB-Var-T1-Easy-Hard";
# What to call to run sequences
my $BEOQUEUE       = "/lab/mundhenk/linear-classifier/script/beoqueue.pl";
# List of nodes to use
my $NODES          = "ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo icore icore icore icore icore icore icore icore icore icore icore icore icore icore icore icore ";
# Perl script to run inside Beoqueue
my $PROC_BASE      = "/lab/mundhenk/linear-classifier/script/process_rsvp_special.pl";
# Perl script to run inside Beoqueue
my $PROC           = "/lab/mundhenk/linear-classifier/script/process_rsvp_special_2.pl";
# What is the output file called?
my $FILE_NAME      = "chan.txt.final-AGmask.txt";
# What to call the logfile
my $LOG_FILE       = "$OUT_DIR/logfile.txt"; 
# Should we run ezvision or do we already have the data?
my $RUN_EZVISION   = 0;
# Should we run the post analysis, if not then we should be running ezvision
my $RUN_ANALYSIS   = 1;
# What column is the mean in?
my $MEAN_COLUMN    = 12;
# What column is the std in?
my $STD_COLUMN     = 13;


if(($RUN_EZVISION == 0) && ($RUN_ANALYSIS == 0))
{
    die "ERROR : Either RUN_EZVISION or RUN_ANALYSIS or both must be set to 1 \(i.e. turned on\) \n";
}

my $A_EASY_DIR  = "$OUT_DIR/A_EASY";
my $B_EASY_DIR  = "$OUT_DIR/B_EASY";
my $A_HARD_DIR  = "$OUT_DIR/A_HARD";
my $B_HARD_DIR  = "$OUT_DIR/B_HARD";

# get the initial list of sequences to run
my $command     = "ls -1 $BASE_DIR";
my @baseDirList = qx"$command";
my $stimNum     = 0;
my @stimDir     = {};
my $TARG_A;     my $TARG_B;
my $A_MAX;      my $A_MIN; 
my $B_MAX;      my $B_MIN; 
my $A_MAX_STIM; my $A_MIN_STIM;
my $B_MAX_STIM; my $B_MIN_STIM; 

my $BASE_MEAN_A = 0;
my $BASE_STD_A  = 0;
my $BASE_MEAN_B = 0;
my $BASE_STD_B  = 0;

# make output directories

$command = "mkdir $OUT_DIR";
system("$command");
$command = "mkdir $A_EASY_DIR";
system("$command");
$command = "mkdir $B_EASY_DIR";
system("$command");
$command = "mkdir $A_HARD_DIR";
system("$command");
$command = "mkdir $B_HARD_DIR";
system("$command");

# create new log file
open(LOGFILE, ">$LOG_FILE");

foreach my $base (@baseDirList)
{
    chomp($base);
    # get the components of the dir name and store as 
    # stim parameters
    my @stim    = split(/\_/,$base);
    my $newBase = "$stim[0]\_$stim[3]\_$stim[4]\_$stim[5]";

    # This is a stim directory
    # Name like : stim_AB_Trans-Anims_06_14_019
    if(($stim[0] eq "stim") && ($stim[1] eq "AB"))
    {
	print("\n>>>> RUNNING $base\n\n");
	# run beoqueue on all sequences in the set for this stim also
	# run it on the baseline variant
	if($RUN_EZVISION)
	{
	    # run the base first
	    $command = "rsh -n icore \"$PROC_BASE $BASE_DIR/$base\"";
	    print("\n>>>> COMMAND $command\n\n");
	    system("$command");
	    
	    # run the variants
	    $command = "$BEOQUEUE -n \"$NODES\" $PROC $BASE_DIR/$base/var\_???";
	    print("\n>>>> COMMAND $command\n\n");
	    system("$command");
	    
	}

	# Run the analysis after ezvision to get the new sets
	if($RUN_ANALYSIS)
	{
	    # what frame will stim A and B be in?
	    $TARG_A = $LEAD_IN_FRAMES + $stim[3];
	    $TARG_B = $LEAD_IN_FRAMES + $stim[4];

	    # get the sub-dirs that will contain the raw stats
	    $command = "ls -1 $BASE_DIR/$base";
	    my @dirList = qx"$command";

	    # get the baseline stats for this stim
	    getBaseline($base);

	    # reset the min and max for these new variants
	    $stimNum = 0;
	    @stimDir = {};
	
	    # for each potential stim in the variant list
	    # we check the mean and std and figure out which is max or min
	    foreach my $dir (@dirList)
	    {
		# get each variant
		my @var = split(/\_/,$dir);
		if($var[0] eq "var")
		{
		    $stimDir[$stimNum] = $dir;
		    parseStats($base,$dir);
		    $stimNum++;
		}
	    }
	    # we now have a preliminary list of which variants should be easy/hard
	    # copy the best variants into their easy-hard sets
	    createSets($base);
	}
    }
}
######################################################################
sub getBaseline
{ 
    my $base = $_[0];
    # open the file for this stim
    my $filename = "$BASE_DIR\/$base\/$FILE_NAME";
    open(STATSFILE, "$filename");
    print("OPEN $filename\n");
    
    while(<STATSFILE>)
    {
	chomp();
	#print("$_\n");
	my @line = split(/\t/);


	# figure out which frame we are looking at
	my $frame = $line[1];
	
	# get the min/max for mean and std for both frame A and B
	if($frame == $TARG_A)
	{
	    $BASE_MEAN_A = $line[$MEAN_COLUMN];
	    $BASE_STD_A  = $line[$STD_COLUMN];
	}
	elsif($frame == $TARG_B)
	{
	    $BASE_MEAN_B = $line[$MEAN_COLUMN];
	    $BASE_STD_B  = $line[$STD_COLUMN];
	}
    }  

    # init to the baseline
    $A_MAX = $BASE_STD_A; 
    $A_MIN = $BASE_MEAN_A;
    $B_MAX = $BASE_STD_B;
    $B_MIN = $BASE_MEAN_B;

    # init to -1, if we get this back then none of the variants
    # are better than the baseline
    $A_MIN_STIM = -1;
    $A_MAX_STIM = -1;
    $B_MIN_STIM = -1;
    $B_MAX_STIM = -1;

    close(STATSFILE); 
}

######################################################################
sub parseStats
{ 
    my $base = $_[0];
    my $dir  = $_[1];

    chomp($dir);

    my $TARG_MEAN_A = -1;
    my $TARG_MEAN_B = -1;
    my $TARG_STD_A  = -1;
    my $TARG_STD_B  = -1;
		
    # open the file for this variant
    my $filename = "$BASE_DIR\/$base\/$dir\/$FILE_NAME";
    open(STATSFILE, "$filename");
    print("OPEN $filename\n");
    
    while(<STATSFILE>)
    {
	chomp();
	#print("$_\n");
	my @line = split(/\t/);


	# figure out which frame we are looking at
	my $frame = $line[1];
	
	# get the min/max for mean and std for both frame A and B
	if($frame == $TARG_A)
	{
	    $TARG_MEAN_A = $line[$MEAN_COLUMN];
	    $TARG_STD_A  = $line[$STD_COLUMN];
	}
	elsif($frame == $TARG_B)
	{
	    $TARG_MEAN_B = $line[$MEAN_COLUMN];
	    $TARG_STD_B  = $line[$STD_COLUMN];
	}
    }

    # check for bugs
    if(($TARG_MEAN_A == -1) || ($TARG_MEAN_B == -1) || ($TARG_STD_A == -1) || ($TARG_STD_B == -1))
    {
	die "ERROR Values not assigned from input file $TARG_MEAN_A $TARG_MEAN_B $TARG_STD_A $TARG_STD_B\n";
    }

    # hard frame A (Use Mean Value)
    if($TARG_MEAN_A < $A_MIN)
    {
	$A_MIN      = $TARG_MEAN_A;
	$A_MIN_STIM = $stimNum;
    }

    # easy frame A (Use Std Value)
    if($TARG_STD_A > $A_MAX)
    {
	$A_MAX       = $TARG_STD_A;
	$A_MAX_STIM  = $stimNum;
    }

    # hard frame B (Use Mean Value)
    if($TARG_MEAN_B < $B_MIN)
    {
	# Frame A should be above or equal to baseline
	if($TARG_MEAN_A >= $BASE_MEAN_B)
	{
	    $B_MIN      = $TARG_MEAN_B;
	    $B_MIN_STIM = $stimNum;
	}
    }
    
    # easy frame B (Use Std Value)
    if($TARG_STD_B > $B_MAX)
    {
        # Frame A should be below or equal to baseline
	if($TARG_STD_A <= $BASE_STD_B)
	{
	    $B_MAX       = $TARG_STD_B;
	    $B_MAX_STIM  = $stimNum;
	}
    }
    close(STATSFILE); 
}

######################################################################
# The EASY frames are ones where STD  is as LARGE as possible
# The HARD frames are ones where MEAN is as SMALL as possible
sub createSets
{ 
    my $base    = $_[0];
    open(LOGFILE, ">>$LOG_FILE");
    print(LOGFILE "$base\t$BASE_STD_A\t$BASE_STD_B\t$BASE_MEAN_A\t$BASE_MEAN_B\t");
    print(LOGFILE "$A_MAX\t$B_MAX\t$A_MIN\t$B_MIN\t");
    print(LOGFILE "$A_MAX_STIM\t$B_MAX_STIM\t$A_MIN_STIM\t$B_MIN_STIM\n");
    createEachSet($base,$A_EASY_DIR,$A_MAX_STIM,$A_MAX,$BASE_STD_A);    
    createEachSet($base,$B_EASY_DIR,$B_MAX_STIM,$B_MAX,$BASE_STD_B);
    createEachSet($base,$A_HARD_DIR,$A_MIN_STIM,$A_MIN,$BASE_MEAN_A);
    createEachSet($base,$B_HARD_DIR,$B_MIN_STIM,$B_MIN,$BASE_MEAN_B);
    close(LOGFILE);
}

######################################################################
sub createEachSet
{
    # make directories
    my $base    = $_[0];
    my $dir     = $_[1];
    my $stim    = $_[2];
    my $stimVal = $_[3];
    my $oldVal  = $_[4];

    chomp($dir);
    chomp($base);
    chomp($stimDir[$stim]);

    my $to      = "$dir/$base";
    my $from    = "$BASE_DIR/$base/$stimDir[$stim]";

    # try to create dir
    $command    = "mkdir $to";
    system("$command");
    
    # try to clean out dir
    $command    = "rm -f $to/*";
    system("$command");
    
    # only link out if we found a better set
    if($stim !=  -1) 
    {
	print(LOGFILE "\t$stimDir[$stim] NEW $stimVal OLD $oldVal\n");

	$command    = "ls -1 $from *.png";
	my @files   = qx"$command";

	foreach my $f (@files)
	{
	    if($f =~ "png")
	    {
		chomp($f);
		$command = "ln $from/$f $to/$f";
		#print(LOGFILE "$command\n"); 
		system("$command");
	    }
	}
    }
}
