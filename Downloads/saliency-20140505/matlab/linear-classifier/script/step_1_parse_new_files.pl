# This is the first step to be called after RSVP image sequences have been run
# It will parse both the full saliency as well as channel data
# into combined files.

#Should we skip the "stats" file and only parse channel data?
$CHANNEL_ONLY   = 1;
#Sets us to use one channel file
$ONE_CHAN_FILE  = 0;
# How many lead in frames were used
$LEAD_IN_FRAMES = 10;
# Should we put a header at the top?
$DO_HEADER      = 0;
# Should we only do the basic stats?
$STATS_ONLY     = 0;
# Should we try to import fourier magnatude frequency data
$DO_FREQ        = 0;
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
	}
	else
	{
	    if ($arg =~ /c/) 
	    { 
		$CHANNEL_ONLY = 0;
		print("SETTING CHANNEL_ONLY to $CHANNEL_ONLY\n");
	    }
	    if ($arg =~ /s/) 
	    { 
		$STATS_ONLY   = 1;
		print("SETTING STATS_ONLY to $STATS_ONLY\n");
	    }
	    if ($arg =~ /o/) 
	    { 
		$ONE_CHAN_FILE = 1;
		print("SETTING ONE_CHAN_FILE to $ONE_CHAN_FILE\n");
	    }
	    if ($arg =~ /h/) 
	    { 
		$DO_HEADER = 1;
		print("SETTING DO_HEADER to $DO_HEADER\n");
	    }
	    if ($arg =~ /f/) 
	    { 
		$DO_FREQ = 1;
		print("SETTING DO_FREQ to $DO_FREQ\n");
	    }
	}
    }
}
else
{
    $SET_BASE_DIR = $ARGV[1];
    $SET_OUT_DIR  = $ARGV[0];
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
#What file(s) to output frequency file
$FREQ_OUT       = "freq.combined.$SET_OUT_DIR";
#What file to output channel file
$STATS_OUT      = "stats.combined.$SET_OUT_DIR.txt";
# extension for fourier files
$FREQ_EXT       = ".freq.txt";

# give an index to features so that we can order them.
$feature_label_count = 42;

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
		   "final");

$count = 0;
foreach $fl (@feature_labels)
{
  $feature_label{$fl} = $count;   # give each feature a index number
  $feature_on[$count] = 1;        # turn on or off features into the output file 
  $count++;
}

$feature_label_count = $count - 1;

# clear the feature counts
for($i = 0; $i < $feature_label_count; $i++)
{
    $fcount[$i] = 0;
}

if($DO_FREQ == 1)
{
    for($i = 0; $i < $feature_label_count; $i++)
    {  
	$freqCleared[$i] = 0;
    }
}

#For each stim type with a target
$command     = "ls -1 $BASE_DIR";
@baseFiles   = qx"$command";

# create clean new output files
initFiles();

if($ONE_CHAN_FILE == 1)
{
    oneChanFile();
}
else
{
    multChanFile();
}

######################################################################
# We call this one if all the channel data is in one file
sub oneChanFile()
{

    # go through each directory
    foreach $bf (@baseFiles)
    {
	chomp($bf);
	$thisFile = $bf;
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
		if($bf =~ "_hard_w")       {$conditionType = 1;}
		elsif($bf =~ "_hard_pre")  {$conditionType = 2;}
		elsif($bf =~ "_hard_post") {$conditionType = 3;}
		else                       {$conditionType = 0;}

		getFileName();
		getStim();
		if($CHANNEL_ONLY == 0)
		{
		    parseStats();
		}
		parseChan();
		if($DO_FREQ == 1)
		{
		    parseFreq();
		}
	    }
	}
	else
	{   
	    #print("\tSkipping $thisFile\n"); 
	}
    }
    
    outputChan(); 
}

######################################################################
# We call this one if data from each channel is placed in its own file
sub multChanFile()
{ 
    open(OUTFILE, ">$OUTPUT_DIR\/$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/$CHANNEL_OUT for append\n";
    # go over each feature one at a time
    foreach $fl (@feature_labels)
    {
	# do we use this feature?
	if($feature_on[$feature_label{$fl}] == 1)
	{
	    print("Looking for feature $fl number $feature_label{$fl} \n");
	    $currentFeature = $fl;

	    # print("FEATURE $fl\n");
	    # go through each directory

	    foreach $bf (@baseFiles)
	    {
		chomp($bf);
		$thisFile = $bf;
		# Skip real files  
		if(substr($bf,0,4) eq "stim")
		{
		    # Check if this is a hard set file
		    if($bf =~ "_hard_w")       {$conditionType = 1;}
		    elsif($bf =~ "_hard_pre")  {$conditionType = 2;}
		    elsif($bf =~ "_hard_post") {$conditionType = 3;}
		    else                       {$conditionType = 0;}

		    if(substr($bf,10,1) eq ".")
		    {
			#print("\tSkipping $thisFile\n");
		    }
		    else
		    {
			#print("RUNNING $thisFile\n");
			getFileName();
			getStim();	
			if($STATS_ONLY == 0)
			{
			    parseChanMult();
			    if($DO_FREQ == 1)
			    {
				parseFreq();
			    }
			}
			else
			{
			    parseStats();
			}
		    }
		}
	    }
	    if($DO_FREQ == 1)
	    {
		if($freqSkipped[$currentFeatureNum] == 0)
		{
		    normMatrix(@currentSize);
		    saveMatrix(@currentSize);
		}
	    }
	    reset;
	}
	else
	{
	    print("SKIPPING feature $fl number $feature_label{$fl} \n");
	}
    }
    close(OUTFILE);
}

######################################################################
######################################################################
sub initFiles()
{
    if(($CHANNEL_ONLY == 0) || ($STATS_ONLY == 1)) 
    {
	open(SOUTFILE, ">$OUTPUT_DIR\/$STATS_OUT") or die "Cannot create requested $OUTPUT_DIR\/$STATS_OUT\n";
	if($DO_HEADER == 1)
	{
	    print(SOUTFILE "FILENAME\tSTIMTYPE\tSTIMNUMBER\tFRAME_NUMBER\tFRAME_MS\tFRAME_OFFSET\tXMAX\tYMAX\tXMIN\tYMIN\tMAX\tMIN\tAVG\tSTD\tNPEAKS\tPEAKSUM\n");
	}
	close(SOUTFILE);
    }
    open(SOUTFILE, ">$OUTPUT_DIR\/$CHANNEL_OUT") or die "Cannot create requested $OUTPUT_DIR\/$CHANNEL_OUT\n";
    close(SOUTFILE);
}

######################################################################
sub getFileName()
{
    $fileName = "$BASE_DIR\/$thisFile";
}

######################################################################
sub getStim()
{
    @part1 = split(/m/,$thisFile);
    @part2 = split(/_/,$part1[1]);
    $stimOffset = $part2[0];
    $stimNumber = $part2[1];
    
    # This is -1 since we are counting from 0
    $realTarget = $LEAD_IN_FRAMES + $stimOffset - 1;
}

######################################################################
sub parseStats()
{ 
    open(STATSFILE, "$fileName\/$STAT_FILE") or die "Cannot open requested $fileName\/$STAT_FILE for reading\n";
    open(SOUTFILE, ">>$OUTPUT_DIR\/$STATS_OUT") or die "Cannot open requested $OUTPUT_DIR\/$STATS_OUT for append\n";
    
    $getline = 0;
    $count = 0;

    while(<STATSFILE>)
    {
	if($getline == 0)
	{
	    if(substr($_,0,1) eq "=")
	    {
		$getline = 1;
		@line    = split(/ /);
		$ms      = $line[1];
	    }
	}
	elsif($getline == 1)
	{
	    chomp();
	    $adjTarget = $count - $realTarget;
	    @line = split(/\#/);
	    print(SOUTFILE "$thisFile\t$stimOffset\t$stimNumber\t$count\t$ms\t$adjTarget\t$line[0]\n");
	    $count++;
	    $getline = 0;
	}
    }
    close(SOUTFILE);
    close(STATSFILE);
}

######################################################################
sub parseChan()
{   
    open(STATSFILE, "$fileName\/$CHANNEL_FILE") or die "Cannot open requested $fileName\/$CHANNEL_FILE for reading\n";

    while(<STATSFILE>)
    {
	chomp();
	#print("$_\n");
	@line = split(/\t/);

	if($line[3] == -1) #Sum of scales
	{
	    $frame    = $line[1];
	    $adjFrame = $frame - $realTarget + 5;
	    if($adjFrame < 11)
	    {
		if($adjFrame >= 0)
		{
		    $feature  = $line[4];
		    $fnum     = $feature_label{$feature};
		    if($feature_on[$fnum] == 1)
		    {
			# print only the flanking 5 images
			# Also order output by feature
			if($conditionType == 0)
			{ 
			    $thisline = "$thisFile\t$stimOffset\t$stimNumber\t$frame\t".
				        "$adjFrame\t$feature\t$line[6]\t$line[7]\t$line[8]\t".
				        "$line[9]\t$line[12]\t$line[13]\t";
			}
			else
			{
			    $thisline = "$thisFile\t$stimOffset\t$stimNumber\t$frame\t".
				        "$adjFrame\t$feature\t$line[6]\t$line[7]\t$line[8]\t".
				        "$line[9]\t$line[12]\t$line[13]\t$conditionType\t";
			} 
			push(@{$fnum},$thisline);
			$fcount[$fnum]++;
		    }
		}
	    }
	}
    }
    close(STATSFILE);
}

######################################################################
sub parseChanMult()
{   
    open(STATSFILE, "$fileName\/$CHANNEL_FILE.$currentFeature.txt");
    #print("OPEN $fileName\/$CHANNEL_FILE.$currentFeature.txt \n");

    while(<STATSFILE>)
    {
	chomp();
	#print("$_\n");
	@line = split(/\t/);

	if($line[3] == -1) #Sum of scales
	{ 
	    $frame    = $line[1];
	    $adjFrame = $frame - $realTarget + 5;
	    if($adjFrame < 11)
	    {
		if($adjFrame >= 0)
		{
		    # print only the flanking 5 images
		    # Also order output by feature
		    
		    #min max mean std
		    if($conditionType == 0)
		    { 
			print(OUTFILE  "$thisFile\t$stimOffset\t$stimNumber\t$frame\t".
			               "$adjFrame\t$line[4]\t$line[6]\t$line[7]\t$line[8]\t".
			               "$line[9]\t$line[12]\t$line[13]\n");
		    }
		    else # hard type
		    {
			print(OUTFILE  "$thisFile\t$stimOffset\t$stimNumber\t$frame\t".
			               "$adjFrame\t$line[4]\t$line[6]\t$line[7]\t$line[8]\t".
			               "$line[9]\t$line[12]\t$line[13]\t$conditionType\n");
		    } 
		    
		}
	    }   
	}
    }
    close(STATSFILE); 
}

######################################################################
sub parseFreq()
{   
    $skip = 0;
    open(FREQFILE, "$fileName\/$CHANNEL_FILE.$currentFeature$FREQ_EXT") or $skip = 1;
    #print("OPEN $fileName\/$CHANNEL_FILE.$currentFeature.txt \n");

    if($skip == 0)
    {
	$currentFeatureNum = $feature_label{$currentFeature};

	# have we cleared this frequency matrix yet?
	if($freqCleared[$currentFeatureNum] == 0)
	{
	    $doClear = 1;
	    $freqCleared[$currentFeatureNum] = 1;
	    print("CLEARING $currentFeature $currentFeatureNum \n");
	}
	else
	{
	    $doClear = 0;
	}

	$adjFrame = -1;

	while(<FREQFILE>)
	{	
	    chomp();
	    # header line for this frame?
	    if(substr($_,0,4) eq "stim")
	    {
		$getline = 1;
		# Parse the header line
		@headerline = split(/\t/);
		$frame      = $headerline[1];
		$adjFrame   = $frame - $realTarget + 5;
		$currentRow = 0;
		# clear our storage matrix
		#print("HEADER $currentFeature $feature_label{$currentFeature} $frame\n");
		if($doClear == 1)
		{
		    $currentSize[0]    = $headerline[5]/2 + 1;
		    $currentSize[1]    = $headerline[6];
		    $currentSize[2]    = $currentFeatureNum;
		    clearMatrix(@currentSize);
		    $doClear = 0;
		}

		if($adjFrame < 11)
		{
		    if($adjFrame >= 0)
		    {
			$featureSamples[$currentFeatureNum]++;
		    }
		}
	    }
	    else
	    {
		if($adjFrame < 11)
		{
		    if($adjFrame >= 0)
		    {	    	    
			@row = split(/\t/);
			addRowMatrix(@row);
			$currentRow++;
		    }
		}
	    }
	}
	close(FREQFILE); 
	$freqSkipped[$currentFeatureNum] = 0;
    }
    else
    {
	#print("SKIPPING FREQUENCY $fileName\/$CHANNEL_FILE.$currentFeature.txt$FREQ_EXT\n");
	$freqSkipped[$currentFeatureNum] = 1;
	
    }
	
}

######################################################################
sub outputChan() 
{ 
    open(SOUTFILE, ">$OUTPUT_DIR\/$CHANNEL_OUT") or die "Cannot open requested $OUTPUT_DIR\/$CHANNEL_OUT for append\n";
    for($i = 0; $i < $feature_label_count; $i++)
    {
	if($fcount[$i] > 0)
	{
	    #for($j = 0; $j < $fcount[$i]; $j++)
	    foreach(@{$i})
	    {
		print(SOUTFILE "$_\n");
	    }
	}
    }
    close(SOUTFILE);
}

######################################################################
sub clearMatrix()
{
    $sizeX = $_[0];
    $sizeY = $_[1];
    $feat  = $_[2];
    for($i = 0; $i < $sizeX; $i++)
    {
	for($j = 0; $j < $sizeY; $j++)
	{	
	    $featureMatrix[$feat][$i][$j] = 0;
	    $featureSamples[$feat] = 0;
	}
    }
}

######################################################################
sub addRowMatrix()
{
    for($i = 0; $i < $currentSize[0]; $i++)
    {  
	$featureMatrix[$currentFeatureNum][$i][$currentRow] += $_[$i];
    }
}
	
######################################################################
sub normMatrix()
{
    $sizeX = $_[0];
    $sizeY = $_[1];
    $feat  = $_[2];
    for($i = 0; $i < $sizeX; $i++)
    {
	for($j = 0; $j < $sizeY; $j++)
	{
	    print("$feat - $featureMatrix[$feat][$i][$j] / $featureSamples[$feat]\n");
	    $featureMatrix[$feat][$i][$j] = $featureMatrix[$feat][$i][$j]/$featureSamples[$feat];
	}
    }
}

######################################################################
sub saveMatrix()
{
    
    $sizeX = $_[0];
    $sizeY = $_[1];
    $feat  = $_[2];
    $name  = $feature_labels[$feat];

    open(FOUTFILE, ">$OUTPUT_DIR\/$FREQ_OUT\.$name\.txt") or 
	die "Cannot open requested $OUTPUT_DIR\/$CHANNEL_OUT\.$name\.txt for append\n";

#    for($j = 0; $j < $sizeY; $j++)
#    {
#	for($i = 0; $i < $sizeX; $i++)
#	{
#	    print(FOUTFILE "$featureMatrix[$feat][$i][$j]\t");
#	}
#	print(FOUTFILE "\n");
#    }

    # shift to put zero frequency spectrum in the center
    # and create a mirror image to complete the standard 
    # fourier image

    # Print upper
    for($j = $sizeY/2 - 1; $j > 0; $j--)
    {
	for($i = $sizeX - 1; $i >= 0; $i--)
	{	
	    print(FOUTFILE "$featureMatrix[$feat][$i][$j]\t");
	}
	for($i = 1; $i < $sizeX; $i++)
	{	
	    print(FOUTFILE "$featureMatrix[$feat][$i][($sizeY) - $j]\t");
	}	
	print(FOUTFILE "\n");
    }

    # print mid
    for($i = $sizeX - 1; $i >= 0; $i--)
    {	
	print(FOUTFILE "$featureMatrix[$feat][$i][0]\t");
    } 
    for($i = 1; $i < $sizeX; $i++)
    {	
	print(FOUTFILE "$featureMatrix[$feat][$i][0]\t");
    }
    print(FOUTFILE "\n");

    # print lower
    for($j = $sizeY - 1; $j >= $sizeY/2; $j--)
    {
	for($i = $sizeX - 1; $i >= 0; $i--)
	{	
	    print(FOUTFILE "$featureMatrix[$feat][$i][$j]\t");
	}
	for($i = 1; $i < $sizeX; $i++)
	{	
	    print(FOUTFILE "$featureMatrix[$feat][$i][($sizeY) - $j]\t");
	}	
	print(FOUTFILE "\n");
    }

    close(FOUTFILE);
}
