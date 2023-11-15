# For stims created using make-sequences.pl

# This is the first step to be called after RSVP image sequences have been run
# It will parse both the full saliency as well as channel data
# into combined files.

use FileHandle; # Allow file handles to be treated like perl vars

# How many lead in frames were used
$LEAD_IN_FRAMES = 10;
$SET_BASE_DIR   = "";
$SET_OUT_DIR    = "";
$SET_VAR_DIR    = "";

# Should we only run the AB sequences?
$RUN_AB_ONLY = 1;

# parse command line
if(substr($ARGV[0], 0, 1) eq '-')
{
    while(substr($ARGV[0], 0, 1) eq '-')
    {
	my $arg = shift;	
	chomp($arg);
	if(substr($arg, 0, 2) eq '--')
	{
	    if ($arg =~ /--lead=/) 
	    { 	
		$LEAD_IN_FRAMES = substr($arg, 7, length($arg));
		print("SETTING LEAD_IN_FRAMES to $LEAD_IN_FRAMES\n");
	    }
	    if ($arg =~ /--base=/) 
	    { 
		$SET_BASE_DIR = substr($arg, 7, length($arg)); 
		print("SETTING BASE_DIR to $SET_BASE_DIR\n"); 
	    }
	    if ($arg =~ /--out=/) 
	    { 
		$SET_OUT_DIR = substr($arg, 6, length($arg));  
		print("SETTING SET_OUT_DIR to $SET_OUT_DIR\n");
	    }
	    if ($arg =~ /--var=/) 
	    { 
		$SET_VAR_DIR = substr($arg, 6, length($arg));  
		print("SETTING SET_VAR_DIR to $SET_VAR_DIR\n");
	    }
	}
    }
}

if($SET_BASE_DIR eq "")
{
    die "Must supply a valid base dir using --base=\n";
}
if($SET_OUT_DIR eq "")
{
    die "Must supply a valid out dir using --out=\n";
}
if($SET_VAR_DIR ne "")
{
    print("Optional variant directory set, will parse variant files");
    $DO_VAR = 1;
}
else
{
    $DO_VAR = 0;
}

# This is the basic chanel statistics file
$CHANNEL_FILE   = "chan.txt";         
# This is the basic whole saliency/surprise file
$STAT_FILE      = "stats.txt";
# this is the base directory all the files are in
#$BASE_DIR       = "/lab/raid/images/RSVP/fullSequence";
$BASE_DIR       = $SET_BASE_DIR;
#Where to place the output
#$OUTPUT_DIR     = "/lab/raid/images/RSVP/fullSequence";
$OUTPUT_DIR     = $SET_BASE_DIR;
#What file to output channel file
$CHANNEL_OUT    = "chan.combined.$SET_OUT_DIR.txt";

# give an index to features so that we can order them.
$feature_label_count = 44;

@feature_labels = ("blank",
		   "ori_0","ori_1","ori_2","ori_3",
		   "h1","h2","sat","val",
		   "dir_0","dir_1","dir_2","dir_3",
		   "flicker","rg","by","intensity",
		   "junction_10101010","junction_01010101",
		   "junction_10001010","junction_10100010","junction_10101000","junction_00101010",
		   "junction_01000101","junction_01010001","junction_01010100","junction_00010101",
		   "junction_10100000","junction_00101000","junction_00001010","junction_10000010",
		   "junction_01010000","junction_00010100","junction_00000101","junction_01000001",
		   "junction_10000000","junction_00100000","junction_00001000","junction_00000010",
		   "junction_01000000","junction_00010000","junction_00000100","junction_00000001",
		   "contour",
		   'r-g','g-r','b-y','y-b',
		   "cband_0","cband_1","cband_2","cband_3","cband_4","cband_5",
		   "final","final-lam",'final-AGmask');

my $count = 0;
foreach $fl (@feature_labels)
{
  $feature_label{$fl} = $count;
  $count++;
}

$feature_label_count = $count - 1;

# turn on or off features into the output file 
# each number coorespods to one in the feature_lable hash table
@feature_on = (1,
	       1,1,1,1, 
	       1,1,1,1, 
	       1,1,1,1,
	       1,1,1,1,
	       1,1,
	       1,1,
	       1,1,1,1,
	       1,1,1,1,
	       1,1,1,1,
	       1,1,1,1,
	       1,1,1,1,
	       1,1,1,1,
	       1,
	       1,1,1,1, 
	       1,1,1,1,1,1,
	       1,1,1);

# clear the feature counts
for(my $i = 0; $i < $feature_label_count; $i++)
{
    $fcount[$i] = 0;
}

#For each stim type with a target
my $command  = "ls -1 $BASE_DIR";
@baseFiles   = qx"$command";

$OUTFILE_A = new FileHandle;
$OUTFILE_B = new FileHandle;
$OUTFILE_W = new FileHandle;
$OUTFILE_T = new FileHandle;
$STATSFILE = new FileHandle;

initFiles();

# go over each feature one at a time
foreach $fl (@feature_labels)
{
    # do we use this feature?
    if($feature_on[$feature_label{$fl}] == 1)
    {   
	print("OPEN $OUTPUT_DIR\/A\.$CHANNEL_OUT\n");
	$OUTFILE_A->open(">>$OUTPUT_DIR\/A\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/A\.$CHANNEL_OUT for append\n";
	print("OPEN $OUTPUT_DIR\/B\.$CHANNEL_OUT\n");
	$OUTFILE_B->open(">>$OUTPUT_DIR\/B\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/B\.$CHANNEL_OUT for append\n";
	print("OPEN $OUTPUT_DIR\/W\.$CHANNEL_OUT\n");
	$OUTFILE_W->open(">>$OUTPUT_DIR\/W\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/W\.$CHANNEL_OUT for append\n";
	print("OPEN $OUTPUT_DIR\/T\.$CHANNEL_OUT\n");
	$OUTFILE_T->open(">>$OUTPUT_DIR\/T\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/T\.$CHANNEL_OUT for append\n";

	print("Looking for feature $fl number $feature_label{$fl} \n");
	$currentFeature = $fl;
	
        # go through each directory
	foreach $bf (@baseFiles)
	{
	    chomp($bf);
	    my $thisFile = $bf;
	    # Skip real files  
	    if(substr($bf,0,4) eq "stim")
	    {
		if(substr($bf,10,1) eq ".")
		{
		    #print("\tSkipping $thisFile\n");
		}
		else
		{
		    #print("RUNNING $thisFile\n");
	    
		    # Check if this is a hard set file
		    if(   $bf =~ "_hard_w")    {my $conditionType = 1;}
		    elsif($bf =~ "_hard_pre")  {my $conditionType = 2;}
		    elsif($bf =~ "_hard_post") {my $conditionType = 3;}
		    else                       {my $conditionType = 0;}
		    
		    my $ABType = -1;

		    # What AB type are we running
		    if(   $bf =~ "stim_Anims")          {$ABType = 1;}
		    elsif($bf =~ "stim_Trans")          {$ABType = 2;}
		    elsif($bf =~ "stim_AB_Anims-Trans") {$ABType = 3;} 
		    elsif($bf =~ "stim_AB_Trans-Anims") {$ABType = 4;}
		    else                                {die "Unknown AB type. Cannot determine from directory \'$bf\'\n";}
		    
		    if(($RUN_AB_ONLY == 0) || ($ABType > 2))
		    { 
			#print("RUNNING $thisFile\n");
			getFileName($thisFile);
			getStim($thisFile,$ABType);
			
			parseChanMult($ABType,-1);
			if($DO_VAR)
			{
			    parseChanVar($ABType);
			}	
		    }
		}
	    }
	}
	$OUTFILE_A->close();
	$OUTFILE_B->close();
	$OUTFILE_W->close();
	$OUTFILE_T->close();
    }
    else
    {   
	#print("\tSkipping $thisFile\n"); 
    }


}
    
######################################################################
sub getFileName()
{
    $thisFile = $_[0];
    $fileName = "$BASE_DIR\/$thisFile";
}

######################################################################
sub getStim()
{
    my $thisFile = $_[0];
    my $ABType   = $_[1]; 

    @parts = split(/\_/,$thisFile);
    
    if($ABType < 3)  # Single targets only
    {

	$stimOffset[0] = $parts[2];
	$stimOffset[1] = -1;
	$stimNumber    = $parts[3];
	# This is -1 since we are counting from 0
	$realTarget[0] = $LEAD_IN_FRAMES + $stimOffset[0] - 1;
	$realTarget[1] = -1;
    }
    else             # AB Targets
    {
	$stimOffset[0] = $parts[3];
	$stimOffset[1] = $parts[4];
	$stimNumber    = $parts[5];
	# This is -1 since we are counting from 0
	$realTarget[0] = $LEAD_IN_FRAMES + $stimOffset[0] - 1;
	$realTarget[1] = $LEAD_IN_FRAMES + $stimOffset[1] - 1;
    }
}

######################################################################
sub parseChanVar();
{
    my $ABType = $_[0];
    my $command     = "ls -1 $fileName";
    @varFiles   = qx"$command";
    
    my $count = 0;

    # for each variant within each sequence, process
    foreach $vf (@varFiles)
    {
	if(substr($vf,0,3) eq "$SET_VAR_DIR")
	{
	    parseChanMult($ABType,$count);
	    $count++;
	}
    }
}

######################################################################
sub parseChanMult()
{   
    my $ABType  = $_[0];
    my $variant = $_[1];

    my $opened = 0;

    if($variant == -1)
    {
	$STATSFILE->open("$fileName\/$CHANNEL_FILE.$currentFeature.txt");
	#print("OPEN $fileName\/$CHANNEL_FILE.$currentFeature.txt \n");
	$opened = 1;
    }
    else
    {
	my $vname = sprintf("%03d", $variant);
	$STATSFILE->open("$fileName\/var\_$vname\/$CHANNEL_FILE.$currentFeature.txt"); 
	$opened = 1;
    }

    while(<$STATSFILE>)
    {
	chomp();
	#print("$_\n");
	@line = split(/\t/);

	if(($line[3] == -1) || ($line[3] eq "LAM")) #Sum of scales
	{ 
	    $frame       = $line[1];
	    $adjFrame[0] = $frame - $realTarget[0] + 5;
	    $adjFrame[1] = $frame - $realTarget[1] + 5;

	    # output file with frames centered around target A
	    if($adjFrame[0] < 11)
	    {
		if($adjFrame[0] >= 0)
		{
		    # print only the flanking 5 images
		    # Also order output by feature
		    
		    #min max mean std
		    if($conditionType == 0)
		    { 
			printBase($OUTFILE_A,$adjFrame[0],1,$variant,$ABType);
		    }
		    else # hard type
		    {
			printHard($OUTFILE_A,$adjFrame[0],1,$variant,$ABType);
		    }  
		}
                # save just the target for AB only
		if(($adjFrame[0] == 5) && ($ABType > 2))
		{
		    #min max mean std
		    if($conditionType == 0)
		    { 
			printBase($OUTFILE_T,$adjFrame[0],1,$variant,$ABType);
		    }
		    else # hard type
		    {
			printHard($OUTFILE_T,$adjFrame[0],1,$variant,$ABType);
		    }
		} 
		    
	    }
	    # output file with frames centered around target B if it exists
	    if(($ABType > 2) && ($adjFrame[1] < 11))
	    {
		if($adjFrame[1] >= 0)
		{
		    # print only the flanking 5 images
		    # Also order output by feature
		    
		    #min max mean std
		    if($conditionType == 0)
		    { 
			printBase($OUTFILE_B,$adjFrame[1],2,$variant,$ABType);
		    }
		    else # hard type
		    {
			printHard($OUTFILE_B,$adjFrame[1],2,$variant,$ABType);
		    }  
		}
		# save just the target for AB only
		if($adjFrame[1] == 5)
		{
		    #min max mean std
		    if($conditionType == 0)
		    { 
			printBase($OUTFILE_T,$adjFrame[1],2,$variant,$ABType);
		    }
		    else # hard type
		    {
			printHard($OUTFILE_T,$adjFrame[1],2,$variant,$ABType);
		    }
		} 
		
	    }

	    # Create a Whole file with everything in it
	    if($conditionType == 0)
	    { 
		printBase($OUTFILE_W,0,0,$variant,$ABType);
	    }
	    else # hard type
	    {
		printHard($OUTFILE_W,0,0,$variant,$ABType);
	    }  
	}
    }
    if($opened == 1)
    {
	$STATSFILE->close();
    } 
}

######################################################################
sub printBase
{
    my $file     = $_[0];
    my $adjframe = $_[1];
    my $ttype    = $_[2];  
    my $variant  = $_[3];
    my $ABType   = $_[4];

    $offsetDiff  = $stimOffset[1] - $stimOffset[0];

 
    if($currentFeature eq "final-AGmask")
    {
	$thisChannel = "final-AGmask";
	$file->print("$thisFile\t$stimOffset[0]\t$stimOffset[1]\t$offsetDiff\t".
		     "$stimNumber\t$variant\t$ABType\t$frame\t".
		     "$adjframe\t$ttype\t$thisChannel\t$line[10]\t$line[11]\t$line[12]\t".
		     "$line[13]\t$line[16]\t$line[17]\t$line[8]\n");
    }
    else
    {
	if($currentFeature eq "final-lam")
	{
	    $thisChannel = "final-lam";
	}
	else
	{
	    $thisChannel = $line[4];
	}
	$file->print("$thisFile\t$stimOffset[0]\t$stimOffset[1]\t$offsetDiff\t".
		     "$stimNumber\t$variant\t$ABType\t$frame\t".
		     "$adjframe\t$ttype\t$thisChannel\t$line[6]\t$line[7]\t$line[8]\t".
		     "$line[9]\t$line[12]\t$line[13]\t-1\n");
    }
}

######################################################################
sub printHard
{
    my $file     = $_[0];
    my $adjframe = $_[1];
    my $ttype    = $_[2];
    my $variant  = $_[3];
    my $ABType   = $_[4];

    $offsetDiff  = $stimOffset[1] - $stimOffset[0];
    
    if($currentFeature eq "final-AGmask")
    {
	$thisChannel = "final-AGmask";
	$file->print("$thisFile\t$stimOffset[0]\t$stimOffset[1]\t$offsetDiff\t".
		     "$stimNumber\t$variant\t$ABType\t$frame\t".
		     "$adjframe\t$ttype\t$thisChannel\t$line[10]\t$line[11]\t$line[12]\t".
		     "$line[13]\t$line[16]\t$line[17]\t$conditionType\t$line[8]\n");
    }
    else
    {
	$thisChannel = "final-AGmask";
	if($currentFeature eq "final-lam")
	{
	    $thisChannel = "final-lam";
	} 
	else
	{
	    $thisChannel = $line[4];
	} 
	$file->print("$thisFile\t$stimOffset[0]\t$stimOffset[1]\t$offsetDiff\t".
		     "$stimNumber\t$variant\t$ABType\t$frame\t".
		     "$adjframe\t$ttype\t$thisChannel\t$line[6]\t$line[7]\t$line[8]\t".
		     "$line[9]\t$line[12]\t$line[13]\t$conditionType\t-1\n");
    }
}

######################################################################
sub initFiles
{
    print("INIT $OUTPUT_DIR\/A\.$CHANNEL_OUT\n");
    $OUTFILE_A->open(">$OUTPUT_DIR\/A\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/A\.$CHANNEL_OUT for init\n";
    print("INIT $OUTPUT_DIR\/B\.$CHANNEL_OUT\n");
    $OUTFILE_B->open(">$OUTPUT_DIR\/B\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/B\.$CHANNEL_OUT for init\n";
    print("INIT $OUTPUT_DIR\/W\.$CHANNEL_OUT\n");
    $OUTFILE_W->open(">$OUTPUT_DIR\/W\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/W\.$CHANNEL_OUT for init\n";
    print("INIT $OUTPUT_DIR\/T\.$CHANNEL_OUT\n");
    $OUTFILE_T->open(">$OUTPUT_DIR\/T\.$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/T\.$CHANNEL_OUT for init\n";

    $OUTFILE_A->close();
    $OUTFILE_B->close();
    $OUTFILE_W->close();
    $OUTFILE_T->close();
    print("done\n");
}
