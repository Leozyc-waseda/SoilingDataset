# Give the stim type as an input like 
#
# perl make-sequences.pl -AB 
#
# or we can call a default sequence to create 

if(substr($ARGV[0], 0, 1) eq '-')
{
    my $arg = shift;
    chomp($arg);
    $DOTHIS = substr($arg, 0, length($arg));
}
else
{ 
    $DOTHIS = "AB_PARTIAL_REPETE_VAR_T1";                        # Which kind of stim to make
}

$DEBUG  = 0;

@Anims   = qx"ls -1 ../Anims/*.png";   # List of All animal images
@Trans   = qx"ls -1 ../Trans/*.png";   # List of All transportation images 
@Distras = qx"ls -1 ../Distras/*.png"; # List of All distractors
 
$AnimsNum   = @Anims;                  # Number of Anim Images
$TransNum   = @Trans;                  # Number of Trans Images
$DistrasNum = @Distras;                # Number of Distra Images

@AnimsFile   =  chopFrame(@Anims);     # Get the file name only (strip directory)
@TransFile   =  chopFrame(@Trans);     # Get the file name only (strip directory)
@DistrasFile =  chopFrame(@Distras);   # Get the file name only (strip directory)

$FILELIST_NAME = "filelist.txt";
$STIMLIST_NAME = "stimlist.txt";
$MAX_TRIES     = 5000;                 # How many trys should be used in AB, keeps us from infinite loop
# Overdo the random number seed
srand (time ^ $$ ^ unpack "%L*", `ps axww | gzip -f`);

# Create the file list to translate back if needed
open(FILELIST, ">$FILELIST_NAME");
close(FILELIST);

# Create the stim list for accounting
open(STIMLIST, ">$STIMLIST_NAME");
close(STIMLIST);

######################################################################
if($DOTHIS eq "TRANS")
{
    $NO_TARG           = 500; # How many to target stims are there?
    $BASE_OFFSET       = 6;   # Starting offset for the target
    $NUM_PER_COND      = 50;  # Number of samples per condition
    $CONDS             = 10;  # Number of conditions
    $SEQ_SIZE          = 20;  # Number of frames per sequence
    $PRE_FIXATE        = 10;  # How many fixation frames to lead in with
    $POST_FIXATE       = 2;   # How many post fixation frames to follow with

    $CREATE_N_VARIANTS = 0;   # Should we create N variants?

    for($i = 0; $i < $NO_TARG; $i++)
    {
	createNoTargSequence($i);
    }

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Trans");
	}
    }
}

######################################################################
elsif($DOTHIS eq "ANIMS")
{
    $NO_TARG           = 500; # How many to target stims are there?
    $BASE_OFFSET       = 6;   # Starting offset for the target
    $NUM_PER_COND      = 50;  # Number of samples per condition
    $CONDS             = 10;  # Number of conditions
    $SEQ_SIZE          = 20;  # Number of frames per sequence  
    $PRE_FIXATE        = 10;  # How many fixation frames to lead in with
    $POST_FIXATE       = 2;   # How many post fixation frames to follow with

    $CREATE_N_VARIANTS = 0;   # Should we create N variants?

    for($i = 0; $i < $NO_TARG; $i++)
    {
	createNoTargSequence($i);
    }

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Anims");
	}
    }
}

######################################################################
elsif($DOTHIS eq "AB")
{
    $NO_TARG           = 500; # How many to target stims are there?
    $BASE_OFFSET       = 6;   # Starting offset for the B target
    $NUM_PER_COND      = 25;  # Number of samples per condition
    $CONDS             = 10;  # Number of conditions
    $SEQ_SIZE          = 20;  # Number of frames per sequence
    $BASE_A_OFFSET     = 0;   # What frame should target A fall into relative to $BASE_OFFSET?
    $BASE_B_OFFSET     = 1;   # What frame should target B fall into relative to $BASE_OFFSET?  
    $PRE_FIXATE        = 10;  # How many fixation frames to lead in with
    $POST_FIXATE       = 2;   # How many post fixation frames to follow with
    $VAR_COUNT         = 100; # How many variants to make of each sequence

    $CREATE_N_VARIANTS = 1;   # Should we create N variants?  

    for($i = 0; $i < $NO_TARG; $i++)
    {
	createNoTargSequence($i);
    }

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Trans",-1);
	}
    }

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Anims",-1);
	}
    }

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequenceAB(0,$c,$n,"Anims-Trans",-1,-1);
	}
    }  

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequenceAB(0,$c,$n,"Trans-Anims",-1,-1);
	}
    }
}

######################################################################
# This is AB, but we keep a list of 250 animals / transports in the single
# target conditions and we re-use them in order in the AB conditions. 
elsif($DOTHIS eq "AB_PARTIAL_REPETE")
{
    $NO_TARG           = 500; # How many to target stims are there?
    $BASE_OFFSET       = 6;   # Starting offset for the B target
    $NUM_PER_COND      = 25;  # Number of samples per condition
    $CONDS             = 10;  # Number of conditions
    $SEQ_SIZE          = 20;  # Number of frames per sequence
    $BASE_A_OFFSET     = 0;   # What frame should target A fall into relative to $BASE_OFFSET?
    $BASE_B_OFFSET     = 1;   # What frame should target B fall into relative to $BASE_OFFSET?  
    $PRE_FIXATE        = 10;  # How many fixation frames to lead in with
    $POST_FIXATE       = 2;   # How many post fixation frames to follow with
    $VAR_COUNT         = 100; # How many variants to make of each sequence

    $CREATE_N_VARIANTS = 1;   # Should we create N variants?
  
    for($i = 0; $i < $NO_TARG; $i++)
    {
	createNoTargSequence($i);
    }

    my $Tcount = 0;
    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Trans",$Tcount);
	    $Tcount++
	}
    }

    my $Acount = 0;
    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Anims",$Acount);
	    $Acount++;
	}
    }

    initTargSequenceAB();

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequenceAB(0,$c,$n,"Anims-Trans",$Tcount,$Acount);
	}
    } 

    initTargSequenceAB();

    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequenceAB(0,$c,$n,"Trans-Anims",$Tcount,$Acount);
	}
    }
}

######################################################################
# This is AB, but we keep a list of 250 animals / transports in the single
# target conditions and we re-use them in order in the AB conditions. 
elsif($DOTHIS eq "AB_PARTIAL_REPETE_VAR_T1")
{
    $NO_TARG           = 500; # How many to target stims are there?
    $BASE_OFFSET       = 6;   # Starting offset for the B target
    $NUM_PER_COND      = 25;  # Number of samples per condition
    $CONDS             = 10;  # Number of conditions
    $SEQ_SIZE          = 20;  # Number of frames per sequence
    $BASE_A_OFFSET     = 0;   # What frame should target A fall into relative to $BASE_OFFSET?
    $BASE_B_OFFSET     = 1;   # What frame should target B fall into relative to $BASE_OFFSET?  
    $PRE_FIXATE        = 10;  # How many fixation frames to lead in with
    $POST_FIXATE       = 2;   # How many post fixation frames to follow with
    $VAR_COUNT         = 100; # How many variants to make of each sequence

    $CREATE_N_VARIANTS = 0;   # Should we create N variants?

    for($i = 0; $i < $NO_TARG; $i++)
    {
	createNoTargSequence($i);
    }

    my $Tcount = 0;
    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Trans",$Tcount);
	    $Tcount++
	}
    }

    my $Acount = 0;
    for($c = 0; $c < $CONDS; $c++)
    {
	for($n = 0; $n < $NUM_PER_COND; $n++)
	{
	    createTargSequence($c,$n,"Anims",$Acount);
	    $Acount++;
	}
    }

    initTargSequenceAB();

    $NUM_PER_COND      = 10;  # Number of samples per condition
    $CONDS             = 5;   # Number of conditions

    for($c1 = 0; $c1 < $CONDS; $c1++)
    {   
	for($c2 = $c1; $c2 < $c1 + $CONDS; $c2++)
	{
	    for($n = 0; $n < $NUM_PER_COND; $n++)
	    {
		createTargSequenceAB($c1,$c2,$n,"Anims-Trans",$Tcount,$Acount);	
	    }
	}
    } 

    initTargSequenceAB();

    for($c1 = 0; $c1 < $CONDS; $c1++)
    {   
	for($c2 = $c1; $c2 < $c1 + $CONDS; $c2++)
	{
	    for($n = 0; $n < $NUM_PER_COND; $n++)
	    { 
		createTargSequenceAB($c1,$c2,$n,"Trans-Anims",$Tcount,$Acount);
	    }
	}
    }
}

######################################################################

else
{
    die "\'$DOTHIS\' is not a valid stimulus set to create\n";
} 

close(FILELIST);

######################################################################
######################################################################
### Function Calls
######################################################################
######################################################################

sub createTargSequence
{
    open(FILELIST, ">>$FILELIST_NAME");
    open(STIMLIST, ">>$STIMLIST_NAME");

    $condition     = $_[0] + $BASE_OFFSET; # what offset frame is our target?
    $number        = $_[1];                # what is the number in the sequence?
    $targType      = $_[2];                # What is the target we are using Anims or Trans?
    my $stimCount  = $_[3];                # If we are keeping a target list, which item is this? -1 means don't keep track

    if($targType eq "Trans")
    {
	$targID = 1;
    }
    elsif($targType eq "Anims")
    {
	$targID = 2;
    }
    else
    {
	die "Unknown target type \'$targType\' given for argument\n";
    }

    ### ADD the target

    # try and add random target and check to make sure we don't add the same one twice EVER   

    
    if($targID == 1)  # Transport
    {	
	$finish = 1;
	while($finish) 
	{
	    # pick a random target
	    $targ = int(rand($TransNum));  
	    # the hash is defined and is 1, we already have this image     
	    if((defined $allTransChk{"$TransFile[$targ]"}) && ($allTransChk{"$TransFile[$targ]"} == 1))
	    {
		$finish = 1;
		print("Already have Target $TransFile[$targ] Skipping\n");
	    }
	    else
	    {
		# we found a new frame, add it and set it as added
		$allTransChk{"$TransFile[$targ]"} = 1;
		$finish = 0;

		# if we are keeping a list of all targets, add this one.
		if($stimCount >=0)
		{
		    $TRANS_TARG[$stimCount] = $targ;
		}
	    }
	}
    }
    elsif($targID == 2) # Animals
    {
	$finish = 1;
	while($finish) 
	{
	    # pick a random target
	    $targ = int(rand($AnimsNum));  
	    # the hash is defined and is 1, we already have this image     
	    if((defined $allAnimsChk{"$AnimsFile[$targ]"}) && ($allAnimsChk{"$AnimsFile[$targ]"} == 1))
	    {
		$finish = 1;
		print("Already have Target $AnimsFile[$targ] Skipping\n");
	    }
	    else
	    {
		# we found a new frame, add it and set it as added
		$allAnimsChk{"$AnimsFile[$targ]"} = 1;
		$finish = 0;

		# if we are keeping a list of all targets, add this one.
		if($stimCount >=0)
		{
		    $ANIMS_TARG[$stimCount] = $targ;
		}
	    }
	}
    }	

    ### ADD the distractors

    for($s = 0; $s < $SEQ_SIZE; $s++)  # for each frame in the sequence
    {
	# Check to make sure this is not the target frame (offset - 1)
	if($s != ($condition - 1))
	{
	    $finish = 1;
	    # try and add random dist and check to make sure we don't add the same one twice
	    while($finish)
	    {
		# pick a random image
		$dist = int(rand($DistrasNum));
		# the hash is defined and is 1, we already have this image
		if((defined $distFrameChk{"$DistrasFile[$dist]"}) && ($distFrameChk{"$DistrasFile[$dist]"} == 1))
		{
		    $finish = 1;
		}
		else
		{
		    # we found a new unused image, add it and set it as added
		    $distFrameChk{"$DistrasFile[$dist]"} = 1;
		    $frameName[$s] = $DistrasFile[$dist];
		    $FRAME[$s]     = $Distras[$dist];
		    $finish = 0;
		}
	    }
	}
	else
	{
	    if($targID == 1)
	    {
		# get the target file name
		$FRAME[$s] = $Trans[$targ];
	    } 
	    elsif($targID == 2)
	    {
		# get the target file name
		$FRAME[$s] = $Anims[$targ];
	    } 
	    else 
	    {
		die "Unknown target type\n";
	    }  
	    
            # do some accounting
	    @parts = split(/\//,$FRAME[$s]);
	    print(STIMLIST "T\t$s\t$targ\t$parts[1]\t$parts[2]\n");
	}
    }

    # We have all the frames, now clean out the hash table
    for($s = 0; $s < $SEQ_SIZE; $s++)
    {
	$distFrameChk{"$frameName[$s]"} = 0;
    }
		
    ### Create the sequence

    # make the sequence directory
    $condForm  = sprintf("%02d", $condition);
    $numForm   = sprintf("%03d", $number);
    if($targID == 1)
    {
	$dir = "stim\_Trans\_$condForm\_$numForm";
    }
    elsif($targID == 2)
    {
	# get the target file name
	$dir = "stim\_Anims\_$condForm\_$numForm";
    } 
    else 
    {
	die "Unknown target type\n";
    }

    $command   = "mkdir $dir"; 
    print("\n$command\n");
    system($command);

    # do some accounting
    print(FILELIST "\n\#$dir\n");

    # $command = "ln $fr $dir/stim$condForm\_$numForm\_$frameForm\.png";
    createFrames("$dir/stim\_$condForm\_$numForm");

    if($CREATE_N_VARIANTS)
    {
	createVariants("Base",$condition - 1,0,$dir);
    }

    close(FILELIST); 
    close(STIMLIST);
}

######################################################################
sub initTargSequenceAB
{
    %allTransChk = 0;
    %allAnimsChk = 0;
}

######################################################################
sub createTargSequenceAB
{  
    open(FILELIST, ">>$FILELIST_NAME"); 
    open(STIMLIST, ">>$STIMLIST_NAME");
    $conditionA    = $_[0] + $BASE_OFFSET + $BASE_A_OFFSET; # what offset frame is our target? 
    $conditionB    = $_[1] + $BASE_OFFSET + $BASE_B_OFFSET; # what offset frame is our target?
    $number        = $_[2];                                 # what is the number in the sequence?
    $targType      = $_[3];                                 # What is the target we are using Anims-Trans or Trans-Anims?
    my $TransCount = $_[4];                                 # If we are keeping a target list, which item is this? -1 means don't use
    my $AnimsCount = $_[5];                                 # If we are keeping a target list, which item is this? -1 means don't use

    if($conditionB <= $conditionA)
    {
	die "B must NOT come before A, got A = \'$conditionA\' B = \'$conditionB\'\n";
    }

    if($targType eq "Trans-Anims")
    {
	$targID = 1;
    }
    elsif($targType eq "Anims-Trans")
    {
	$targID = 2;
    }
    else
    {
	die "Unknown target type \'$targType\' given for argument\n";
    }

    ### ADD the A and B targets

    # try and add random target and check to make sure we don't add the same one twice EVER   
    # NOTE: TransCount and AnimsCount are set to the number of anims and trans if we have 
    #       created a smaller set earlier. This then maps from that smaller set. Otherwise
    #       if either is -1 then we use the full list of targets.

    $finish = 1;
    $fcount = 0;
    while($finish) 
    {
	# pick a random target	    
	if($TransCount < 0) { $targ = int(rand($TransNum));   }
	else                { $targ = int(rand($TransCount)); }
	# the hash is defined and is 1, we already have this image     
	if((defined $allTransChk{"$targ"}) && ($allTransChk{"$targ"} == 1))
	{
	    $finish = 1;
	    print("$fcount : Already have Target $targ Skipping\n");
	    # keep track of number of tries so that we can try to detect a failure
	    $fcount++;
	    if($fcount > $MAX_TRIES) {die "Max trans tries exceded\n";}
	}
	else
	{
	    # we found a new frame, add it and set it as added
	    $allTransChk{"$targ"} = 1;
	    $finish = 0;
	}
    }
    if($targID == 1)
    {
	if($TransCount < 0) {$targA = $targ;}
	else                {$targA = $TRANS_TARG[$targ];}
    }
    elsif($targID == 2)
    {
	if($TransCount < 0) {$targB = $targ;}
	else                {$targB = $TRANS_TARG[$targ];}
    }
    
    $finish = 1;
    $fcount = 0;
    while($finish) 
    {
	# pick a random target
	if($AnimsCount < 0) { $targ = int(rand($AnimsNum));   }
	else                { $targ = int(rand($AnimsCount)); }
	# the hash is defined and is 1, we already have this image     
	if((defined $allAnimsChk{"$targ"}) && ($allAnimsChk{"$targ"} == 1))
	{
	    $finish = 1;
	    print("$fcount : Already have Target $targ Skipping\n");
            # keep track of number of tries so that we can try to detect a failure
	    $fcount++;
	    if($fcount > $MAX_TRIES) {die "Max anims tries exceded\n";}
	}
	else
	{
	    # we found a new frame, add it and set it as added
	    $allAnimsChk{"$targ"} = 1;
	    $finish = 0;
	}
    }
    if($targID == 1)
    {
	if($AnimsCount < 0) { $targB = $targ; }
	else                { $targB = $ANIMS_TARG[$targ];}
    }
    elsif($targID == 2)
    {
	if($AnimsCount < 0) { $targA = $targ; }
	else                { $targA = $ANIMS_TARG[$targ];}
    }	

    ### ADD the distractors

    for($s = 0; $s < $SEQ_SIZE; $s++)  # for each frame in the sequence
    {
	# Check to make sure this is not the target frame (offset - 1)
	if(($s != ($conditionA - 1)) && ($s != ($conditionB - 1))) 
	{
	    $finish = 1;
	    # try and add random dist and check to make sure we don't add the same one twice
	    while($finish)
	    {
		# pick a random image
		$dist = int(rand($DistrasNum));
		# the hash is defined and is 1, we already have this image
		if((defined $distFrameChk{"$DistrasFile[$dist]"}) && ($distFrameChk{"$DistrasFile[$dist]"} == 1))
		{
		    $finish = 1;
		}
		else
		{
		    # we found a new unused image, add it and set it as added
		    $distFrameChk{"$DistrasFile[$dist]"} = 1;
		    $frameName[$s] = $DistrasFile[$dist];
		    $FRAME[$s]     = $Distras[$dist];
		    $finish = 0;
		}
	    }
	}
	elsif($s == ($conditionA - 1)) # This is the A frame
	{
	    if($targID == 1)
	    {
		# get the Trans target file name
		$FRAME[$s] = $Trans[$targA];
	    } 
	    elsif($targID == 2)
	    {
		# get the Anims target file name
		$FRAME[$s] = $Anims[$targA];
	    } 
	    else 
	    {
		die "Unknown target type\n";
	    }

	    # do some accounting
	    @parts = split(/\//,$FRAME[$s]);
	    print(STIMLIST "A\t$s\t$targA\t$parts[1]\t$parts[2]\n");
	}
	elsif($s == ($conditionB - 1)) # This is the B frame
	{
	    if($targID == 1)
	    {
		# get the Anims target file name
		$FRAME[$s] = $Anims[$targB];
	    } 
	    elsif($targID == 2)
	    {
		# get the Trans target file name
		$FRAME[$s] = $Trans[$targB];
	    } 
	    else 
	    {
		die "Unknown target type\n";
	    }
 
            # do some accounting
	    @parts = split(/\//,$FRAME[$s]);
	    print(STIMLIST "B\t$s\t$targB\t$parts[1]\t$parts[2]\n");
	}
	else
	{
	    die "Unknown condition\n";
	}
    }

    # We have all the frames, now clean out the hash table
    for($s = 0; $s < $SEQ_SIZE; $s++)
    {
	$distFrameChk{"$frameName[$s]"} = 0;
    }
		
    ### Create the sequence

    # make the sequence directory
    $condFormA  = sprintf("%02d", $conditionA);
    $condFormB  = sprintf("%02d", $conditionB);
    $numForm    = sprintf("%03d", $number);
    if($targID == 1)
    {
	$dir = "stim\_AB\_Trans-Anims\_$condFormA\_$condFormB\_$numForm";
    }
    elsif($targID == 2)
    {
	# get the target file name
	$dir = "stim\_AB\_Anims-Trans\_$condFormA\_$condFormB\_$numForm";
    } 
    else 
    {
	die "Unknown target type\n";
    }

    $command   = "mkdir $dir"; 
    print("\n$command\n");
    system($command);

    # do some accounting
    print(FILELIST "\n\#$dir\n");

    # $command = "ln $fr $dir/stim$\_$condFormA\_$condFormB\_$numForm\_$frameForm\.png
    createFrames("$dir/stim\_$condFormA\_$condFormB\_$numForm"); 

    if($CREATE_N_VARIANTS)
    {
	createVariants("AB",$conditionA - 1,$conditionB - 1,$dir);
    } 
    close(FILELIST);
    close(STIMLIST);
}

######################################################################
sub createNoTargSequence
{   
    open(FILELIST, ">>$FILELIST_NAME");
    $number = $_[0];

    ### ADD the distractors

    for($s = 0; $s < $SEQ_SIZE; $s++)  # for each frame in the sequence
    {
	# Check to make sure this is not the target frame

	$finish = 1;
	# try and add random dist and check to make sure we don't add the same one twice
	while($finish)
	{
	    # pick a random image
	    $dist = int(rand($DistrasNum));
	    # the hash is defined and is 1, we already have this image
	    if((defined $distFrameChk{"$DistrasFile[$dist]"}) && ($distFrameChk{"$DistrasFile[$dist]"} == 1))
	    {
		$finish = 1;
	    }
	    else
	    {
		# we found a new unused image, add it and set it as added
		$distFrameChk{"$DistrasFile[$dist]"} = 1;
		$frameName[$s] = $DistrasFile[$dist];
		$FRAME[$s]     = $Distras[$dist];
		$finish = 0;
	    }
	}
    }

    # We have all the frames, now clean out the hash table
    for($s = 0; $s < $SEQ_SIZE; $s++)
    {
	$distFrameChk{"$frameName[$s]"} = 0;
    }

    ### Create the sequence

    # make the sequence directory
    $numForm   = sprintf("%03d", $number);
    $dir       = "noTarg\_$numForm";
    $command   = "mkdir $dir"; 
    print("\n$command\n");
    system($command);

    # do some accounting
    print(FILELIST "\n\#$dir\n");

    createFrames("$dir/stim");    
    close(FILELIST);
}

######################################################################
sub createFrames
{  
    $prefix = $_[0];

    # copy the image to the directory 
    $frameCount = 0;

    # Create lead in frames
    for($nn = 0; $nn < $PRE_FIXATE; $nn++)
    {
	$frameForm = sprintf("%06d", $frameCount);
	$command = "ln ../fixate.png $prefix\_$frameForm\.png";
	if($DEBUG)
	{
	    print("\t$command\n");
	}
	system($command); 
	
        # do some accounting
	print(FILELIST "../fixate.png $prefix\_$frameForm\.png\n");
	$frameCount++;
    }

    # Create Actual Frames
    foreach $fr (@FRAME)
    {
	
	$frameForm = sprintf("%06d", $frameCount);
	$command   = "ln $fr $prefix\_$frameForm\.png";
	if($DEBUG)
	{
	    print("\t$command\n");
	}
	system($command);
	
        # do some accounting
	print(FILELIST "$fr $prefix\_$frameForm\.png\n");
	$frameCount++;
    }  

    # Create Lead Out frames
    for($nn = 0; $nn < $POST_FIXATE; $nn++)
    {
	$frameForm = sprintf("%06d", $frameCount);
	$command = "ln ../fixate.png $prefix\_$frameForm\.png";
	if($DEBUG)
	{
	    print("\t$command\n");
	}
	system($command); 
 
	# do some accounting
	print(FILELIST "../fixate.png $prefix\_$frameForm\.png\n");
	$frameCount++;
    }
}

######################################################################
sub createVariants
{
    $vtype = $_[0];
    if($vtype eq "AB")
    {	
	$vframeA = $_[1];
	$vframeB = $_[2];
    }
    elsif($vtype eq "Base")
    {
	$vframeA = $_[1];
	$vframeB = -1;
    }
    else
    {
	die "Unknown condition variant \'$vtype\' given \n";
    }

    $baseDir = $_[3]; 
    
    @OLD_FRAME = @FRAME;
    
    # open and close FILELIST to cause createFrames to write to a different file
    
    close(FILELIST);

    open(FILELIST, ">>$baseDir/varlist.txt");

    # create $VAR_COUNT variants of this frame sequence
    for($vc = 0; $vc < $VAR_COUNT; $vc++)
    {
	$vfinish = 1;
	# do this until we get a unique varaint
	while($vfinish)
	{
	    # reserve the target frame slots
	    $vFrameChk{"$vframeA"} = 1;
	    $vFrameChk{"$vframeB"} = 1;
	    
	    $vframeCount = 0;

	    # for each frame, place it somewhere else randomly
	    foreach $fr (@OLD_FRAME)
	    {
		# check to make sure this is not a target frame
		if(($vframeCount != $vframeA) && ($vframeCount != $vframeB))
		{
		    $finish = 1;
		    # try and add random Distros and check to make sure we don't add the same one twice
		    while($finish)
		    {
			# pick a random frame to remap to
			$vtarg = int(rand($SEQ_SIZE));
			# the hash is defined and is 1, we already have this position in use
			if((defined $vFrameChk{"$vtarg"}) && ($vFrameChk{"$vtarg"} == 1))
			{
			    $finish = 1;
			}
			else
			{
			    # we found a new unused slot, map it and set it as added
			    $vFrameChk{"$vtarg"}    = 1;
			    $FRAME[$vtarg]          = $fr;
			    $frameMap[$vtarg]       = $vframeCount;
			    $finish = 0;
			}
		    }
		}
		else
		{
		    # just copy the target frame(s), don't remap them
		    $FRAME[$vframeCount]    = $fr;
		    $frameMap[$vframeCount] = $vframeCount;
		}
		$vframeCount++;
	    } 
	    
	    # blank out the small frame hash map for re-use
	    for($vh = 0; $vh < $vframeCount; $vh++)
	    {
		$vFrameChk{"$vh"} = 0;
	    }
	    
	    # form a hash key from this sequence
	    $seqHashKey = "";
	    for($vh = 0; $vh < $vframeCount; $vh++)
	    {
		$seqHashKey = "$seqHashKey$frameMap[$vh]";
	    }
	    
	    # check to see if we already have this sequence in the full sequence hash map
	    if((defined $HashFrameChk{"$seqHashKey"}) && ($HashFrameChk{"$seqHashKey"} == 1))
	    {
		print("$seqHashKey FOUND\n");
		$vfinish = 1;                # Not done, sequence exists
	    }
	    else
	    {
		print("$seqHashKey OK\n");
		$vfinish = 0;                # This sequence is unique, done!
		$HashKey[$vc] = $seqHashKey; # record the hash key so we can delete it from the hash map later
		$HashFrameChk{"$seqHashKey"} = 1; # Set this key as found
	    }
	}

	### Create the sequence

	# make the sequence directory
	$varForm    = sprintf("%03d", $vc);
	$dir = "$baseDir/var_$varForm";
	
	$command   = "mkdir $dir"; 
	print("\n$command\n");
	system($command);
	
	# do some accounting
	print(FILELIST "\n\#$dir\n");
	
	# $command = "ln $fr $dir/stim$\_$condFormA\_$condFormB\_$numForm\_$frameForm\.png
	createFrames("$dir/stim");
    }
    
    # blank out the bigger hash map
    foreach $hk (@HashKey)
    {
	$HashFrameChk{"$hk"} = 0;
    }

    # open and close FILELIST to cause createFrames to write to a different file
    
    close(FILELIST);

    open(FILELIST, ">>$FILELIST_NAME");
}
		

######################################################################

sub chopFrame
{
    $count = 0;
    foreach $fr (@_)
    {
	chomp($fr);
	@file = split(/\//,$fr);
	$chopFrame[$count] = $file[2];
	#print("GOT $chopFrame[$count]\n");
	$count++;
    }

    return @chopFrame;
}
