# This is the first step to be called after RSVP image sequences have been run
# It will parse both the full saliency as well as channel data
# into combined files.

# This is the basic chanel statistics file
$CHANNEL_FILE   = "chan.txt";         
# This is the basic whole saliency/surprise file
$STAT_FILE      = "stats.txt";
# this is the base directory all the files are in
$BASE_DIR       = $ARGV[0];
#$BASE_DIR       = "/lab/raid/images/RSVP/fullSequence";
#Where to place the output
$OUTPUT_DIR     = $ARGV[0];
#$OUTPUT_DIR     = "/lab/raid/images/RSVP/fullSequence";


#For each stim type with a target
$command     = "ls -1 $BASE_DIR";
@baseFiles   = qx"$command";

# go through each directory
while(<@baseFiles>)
{
    chomp();
    $thisFile = $_;
    # Skip real files
    #if($thisFile =~ "stim")
    if(substr($_,0,4) eq "stim")
    {
	#if(($thisFile =~ "tbz") || ($thisFile =~ "txt") || ($thisFile =~ "noTgt") || ($thisFile =~ "tgz") )
	if(substr($_,10,1) eq ".")
	{
	    #print("\tSkipping $thisFile\n");
	}
	else
	{
	    getFileName();
	    cleanDir();
	}
    }
    else
    {   
	#print("\tSkipping $thisFile\n"); 
    }
}

######################################################################
sub getFileName()
{
    $fileName = "$BASE_DIR\/$thisFile";
}

######################################################################
sub cleanDir()
{ 
    $command = "rm -f $fileName\/$CHANNEL_FILE&"; 
    #print("CLEANING $command\n");
    #system("$command");
    exec("$command");
    #open(OUTFILE, ">$fileName\/$CHANNEL_FILE") or die "Cannot overight $fileName\/$CHANNEL_FILE\n";
}
