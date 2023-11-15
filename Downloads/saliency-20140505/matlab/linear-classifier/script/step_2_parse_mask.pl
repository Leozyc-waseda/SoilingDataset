# This is an optional second step to call to get the mask image frames

# How many lead in frames were used
$LEAD_IN_FRAMES = 10;

# Are we using full mask images with ground truth? 
$USE_FULL_MASK = 0;

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
	if ($arg =~ /--use_full_mask/)
	{
	    $USE_FULL_MASK = 1;
	    print("USING full mask sequences\n");
	}
    }
}

# this is the base directory all the files are in
#$BASE_DIR       = "/lab/raid/images/RSVP/fullSequence";
$BASE_DIR       = $SET_BASE_DIR;
#Where to place the output
#$OUTPUT_DIR     = "/lab/raid/images/RSVP/fullSequence";
$OUTPUT_DIR     = $SET_BASE_DIR;
#What is the file called?
$FILE_NAME      = "chan.txt.final-AGmask.txt";
#What file to output channel file
$MASK_OUT       = "mask.$SET_OUT_DIR.txt";

#For each stim type with a target
$command     = "ls -1 $BASE_DIR";
@baseFiles   = qx"$command";

# create clean new output files
initFiles();

maskFiles();

# copy the output mask images to a common location
if($USE_FULL_MASK)
{
    copyTargImages();
}

######################################################################
# We call this to go over each mask file
sub maskFiles()
{    
    open(OUTFILE, ">$OUTPUT_DIR\/$MASK_OUT") or die "Cannot open requested $OUTPUT_DIR\/$MASK_OUT for append\n";
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
		getStim();
		parseStats();
	    }
	}
	else
	{   
	    #print("\tSkipping $thisFile\n"); 
	}
    }
}

######################################################################
sub parseStats()
{ 
    $filename = "$BASE_DIR\/$thisFile\/$FILE_NAME";

    open(STATSFILE, "$filename");
    print("OPEN $filename\n");

    while(<STATSFILE>)
    {
	chomp();
	#print("$_\n");
	@line = split(/\t/);

	if($line[3] eq "AG-MASK") #Sum of scales
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
		    print(OUTFILE  "$thisFile\t$stimOffset\t$stimNumber\t$frame\t".
			           "$adjFrame\t$line[6]\t$line[7]\t$line[8]\t$line[9]\t");

		    # Print LAM image statistics
		    print(OUTFILE  "$line[10]\t$line[11]\t$line[12]\t$line[13]\t".
			           "$line[14]\t$line[15]\t$line[16]\t$line[17]\t$line[18]\n");

		}
	    }   
	}
    }
    close(STATSFILE); 
}
######################################################################
sub copyTargImages()
{
    # make directories for hard and easy sequences
    $command = "mkdir $BASE_DIR\/$SET_OUT_DIR\_Hard";
    system("$command");

    $command = "mkdir $BASE_DIR\/$SET_OUT_DIR\_Easy";
    system("$command");

    # Open the list of hard and easy files
    open(HARDEASY, "$BASE_DIR\/Short_hard_easy_list.txt") or die("Cannot open $BASE_DIR\/Short_hard_easy_list.txt for reading\n");

    while(<HARDEASY>)
    {
	chomp();
	$line = $_;
	if(substr($line, 0, 1) eq "#")           # Give the condition : easy or hard
	{
	    $stimType = substr($line, 1, 4);
	}
	elsif(substr($line, 0, 4) eq "stim")     # Give a stim name and number
	{
	    @bits  = split(/\t/,$line);
	    @name  = split(/\_/,$bits[0]);
	    $frame = $LEAD_IN_FRAMES + $name[1]; # Is the image in frame 19 etc?

	    $frameForm = sprintf("%06d",$frame); # Zero pad the frame number 

	    $stimDir    = "stim$name[1]\_$name[2]";
	    $imgName    = "stim-AG-LMASK$frameForm\.png";
	    $newImgName = "stim-AG-LMASK$stimDir\.png"; 

	    $command = "rm -f $BASE_DIR\/$SET_OUT_DIR\_$stimType\/$newImgName"; #remove the old image if its there
	    system("$command");

	    $command = "cp $BASE_DIR\/$stimDir\/$imgName $BASE_DIR\/$SET_OUT_DIR\_$stimType\/$newImgName";
	    print("Copy: $command\n");
	    system("$command");
	}
    }
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
sub initFiles()
{
    open(SOUTFILE, ">$OUTPUT_DIR\/$MASK_OUT") or die "Cannot create requested $OUTPUT_DIR\/$MASK_OUT\n";
    close(SOUTFILE);
}
