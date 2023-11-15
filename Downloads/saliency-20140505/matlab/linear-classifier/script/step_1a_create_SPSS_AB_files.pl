# for sqrt function
use Math::Complex;
# Only give this channel : use "all" for all
#$CHANNEL        = "final";
$CHANNEL        = "final";
#$CHANNEL        = "h2";
# File created by step_1_parse_AB_files.pl
$SURPRISE_DATA    = "/lab/tmpib/u/nathan/newSequences/Partial-Repete-AB-Var-T1/T.chan.combined.base.txt";
# File created by step_1_parse_AB_files.pl
$T1_SURPRISE_DATA = "/lab/tmpib/u/nathan/newSequences/Partial-Repete-AB-Var-T1/A.chan.combined.base.txt";
# File created by step_1_parse_AB_files.pl
$T2_SURPRISE_DATA = "/lab/tmpib/u/nathan/newSequences/Partial-Repete-AB-Var-T1/B.chan.combined.base.txt";
# subject results created by merge_subject_data.pl
$SUBJECT_DATA  = "/home/storage/research/AnimTransDistr/Partial-Repete-AB-Var-T1.subject_results/subject_resp.txt";
# output file for SPSS
$SPSS_DATA     = "surprise_resp\.$CHANNEL\.txt"; 
# type of run "T1T2" "T1" or "T2"
$RUN_TYPE      = "T2";
# step one: We parse in and store the subject data since it's smaller
#           we will then store this indexed.

open(SUBJECT, "$SUBJECT_DATA") or die"Connot open subject data \"$SUBJECT_DATA\" for reading\n";

my %columns;

while(<SUBJECT>)
{  
    chomp();
    @parts = split(/,/); # comma delim.
   
    # parse header
    if($line == 0)
    {
	my $count = 0;
	foreach (@parts)
	{
	    $columns{"$_"} = $count;
	    $count++;
	}
    }
    else
    {
	my $stimType = -1;

	if($parts[2] == 2) # AB type
	{
	    if($parts[3] eq "Anims-Trans")
	    {
		$stimType = 1;
	    }
	    elsif($parts[3] eq "Trans-Anims")
	    {
		$stimType = 2;
	    }
	    else
	    {
		die "Unknown AB type given as \"$parts[3]\"\n";
	    }
	}
	else
	{
	    $stimType = 0;
	}
	
	# only parse AB files
	if($stimType > 0)
	{
	    $stimNum  = $parts[$columns{"STIM"}];
	    $tframeA  = $parts[$columns{"TFRAME_A"}];
	    $tframeB  = $parts[$columns{"TFRAME_B"}];
	    $posA     = $parts[$columns{"POS_A"}];
	    $posB     = $parts[$columns{"POS_B"}];
	    $posA_Avg = $parts[$columns{"AVG_POS_A"}];
	    $posB_Avg = $parts[$columns{"AVG_POS_B"}];

	    # store the reponses in an array offset by stim number of target frame number
	    $respT1[$stimType][$stimNum][$tframeA][$tframeB]    = $posA;
	    $respT2[$stimType][$stimNum][$tframeA][$tframeB]    = $posB;
	    $respT1Avg[$stimType][$stimNum][$tframeA][$tframeB] = $posA_Avg;
	    $respT2Avg[$stimType][$stimNum][$tframeA][$tframeB] = $posB_Avg;
	}
    }
    $line++;
}
close(SUBJECT);	
if($RUN_TYPE eq "T1T2")
{
    createT1T2Files();
}
elsif($RUN_TYPE eq "T1")
{
    createT1orT2Files("T1")
}
elsif($RUN_TYPE eq "T2")
{
    createT1orT2Files("T2")
}
else
{
    die "Unknown option given for RUN_TYPE \"$RUN_TYPE\" \n";
}

######################################################################
sub createT1orT2Files
{
    $TType = $_[0];

    open(SPSS,     ">$TType\.$SPSS_DATA")   or die"Cannot open spss data \"$TType\.$SPSS_DATA\" for writing\n"; 
    if($TType eq "T1")
    {
       open(SURPRISE, "$T1_SURPRISE_DATA")  or die"Cannot open surprise data \"$T1_SURPRISE_DATA\" for reading\n";
    }
    elsif($TType eq "T2")
    {
	open(SURPRISE, "$T2_SURPRISE_DATA") or die"Cannot open surprise data \"$T2_SURPRISE_DATA\" for reading\n";	
    }
    else
    {
	die "Unknown target type given as \"$TType\" \n";
    }

    print(SPSS "NAME,POS_A,POS_B,AVG_POSA,AVG_POSB,DAVGPOSA,DAVGPOSB,DIFFPOSA,DIFFPOSB,DIFF_POS,DDIF_POS,TFRAME_A,TFRAME_B,TFRAME_DIFF,");
    print(SPSS "CHANNEL,FRAME,NEW_FRAME,MIN,MAX,AVG,STD,MAXX,MAXY,AREA\n");
    while(<SURPRISE>)
    {
	chomp();
	@parts = split(/\t/); # Tab delim

	# select for channel
	if(($CHANNEL eq "all") || ($parts[9] eq "$CHANNEL"))
	{
	    $targ  = $parts[7];   # T1 or T2;
	    if($targ == 0) # T1
	    {	
		@nameParts = split(/\_/,$parts[0]);
		
		$tframeA = @nameParts[3];
		$tframeB = @nameParts[4]; 
		$stimNum = @nameParts[5];
		
		if($nameParts[2] eq "Anims-Trans")
		{
		    $stimType = 1; 
		}
		elsif($nameParts[2] eq "Trans-Anims")
		{
		    $stimType = 2; 
		}
		else
		{
		    die "Unknown AB type given as \"$nameParts[2]\"\n";
		}
		
		$posA     = $respT1[$stimType][$stimNum][$tframeA][$tframeB];	    
		if($posA    == 5) { $diffPosA = 3; } # easy
		elsif($posA == 1) { $diffPosA = 1; } # hard
		else              { $diffPosA = 2; } # med
		
		$posA_Avg = $respT1Avg[$stimType][$stimNum][$tframeA][$tframeB]; 
		if($posA_Avg    >= 5) { $diffPosA_Avg = 1; } # hard
		elsif($posA_Avg <= 2) { $diffPosA_Avg = 3; } # easy
		else                  { $diffPosA_Avg = 2; } # med
		
		$posB     = $respT2[$stimType][$stimNum][$tframeA][$tframeB];		
		if($posB    == 5) { $diffPosB = 3; } # easy
		elsif($posB == 1) { $diffPosB = 1; } # hard
		else              { $diffPosB = 2; } # med	 
		
		$posB_Avg = $respT2Avg[$stimType][$stimNum][$tframeA][$tframeB];
		if($posB_Avg    >= 5) { $diffPosB_Avg = 1; } # hard
		elsif($posB_Avg <= 2) { $diffPosB_Avg = 3; } # easy
		else                  { $diffPosB_Avg = 2; } # med
		
		$diffPos = $posA - $posB;
		if($diffPos    >=  4) { $descDiffPos = 3; }
		elsif($diffPos <= -4) { $descDiffPos = 1; }
		else                  { $descDiffPos = 2; }
	    }

	    print(SPSS "$parts[0],$posA,$posB,$posA_Avg,$posB_Avg,$diffPosA_Avg,$diffPosB_Avg,".
		       "$diffPosA,$diffPosB,$diffPos,$descDiffPos,".
		       "$parts[1],$parts[2],$parts[3],$parts[9],");

	    $MIN  = $parts[10];
	    $MAX  = $parts[11];
	    $AVG  = $parts[12];
	    $STD  = $parts[13];
	    $MAXX = $parts[14];
	    $MAXY = $parts[15];
	    $AREA = $parts[16];

	    print(SPSS "$parts[6],$parts[7],$MIN,$MAX,$AVG,$STD,$MAXX,$MAXY,$AREA\n");
	}
    }
    close(SURPRISE);
    close(SPSS);
}

######################################################################
sub createT1T2Files
{
    open(SPSS,     ">$SPSS_DATA")    or die"Cannot open spss data \"$SPSS_DATA\" for writing\n"; 
    open(SURPRISE, "$SURPRISE_DATA") or die"Cannot open surprise data \"$SURPRISE_DATA\" for reading\n";
    
    print(SPSS "NAME,POS_A,POS_B,AVG_POSA,AVG_POSB,DAVGPOSA,DAVGPOSB,DIFFPOSA,DIFFPOSB,DIFF_POS,DDIF_POS,TFRAME_A,TFRAME_B,TFRAME_DIFF,CHANNEL,".
	       "T1_MIN,T1_MAX,T1_AVG,T1_STD,T1_MAXX,T1_MAXY,T1_AREA,BR1,".
	       "T2_MIN,T2_MAX,T2_AVG,T2_STD,T2_MAXX,T2_MAXY,T2_AREA,BR2,".
               "DIFF_MIN,DIFF_MAX,DIFF_AVG,DIFF_STD,DIFF_MAXX,DIFF_MAXY,DIFF_AREA,DIFF_DIST,BR3,".
               "EIG_MIN,EIG_MAX,EIG_AVG,EIG_STD,EIG_MAXX,EIG_MAXY,EIG_AREA".
               "\n");  

    while(<SURPRISE>)
    {
	chomp();
	@parts = split(/\t/); # Tab delim
	$targ  = $parts[8];   # T1 or T2;
	
	# select for channel
	if(($CHANNEL eq "all") || ($parts[9] eq "$CHANNEL"))
	{
	    #$print("$CHANNEL\n");
	    if($targ == 1) # T1
	    {
		@nameParts = split(/\_/,$parts[0]);
		
		$tframeA = @nameParts[3];
		$tframeB = @nameParts[4]; 
		$stimNum = @nameParts[5];
		
		if($nameParts[2] eq "Anims-Trans")
		{
		    $stimType = 1; 
		}
		elsif($nameParts[2] eq "Trans-Anims")
		{
		    $stimType = 2; 
		}
		else
		{
		    die "Unknown AB type given as \"$nameParts[2]\"\n";
		}
		
		$posA     = $respT1[$stimType][$stimNum][$tframeA][$tframeB];	    
		if($posA    == 5) { $diffPosA = 3; } # easy
		elsif($posA == 1) { $diffPosA = 1; } # hard
		else              { $diffPosA = 2; } # med
		
		$posA_Avg = $respT1Avg[$stimType][$stimNum][$tframeA][$tframeB]; 
		if($posA_Avg    >= 5) { $diffPosA_Avg = 1; } # hard
		elsif($posA_Avg <= 2) { $diffPosA_Avg = 3; } # easy
		else                  { $diffPosA_Avg = 2; } # med
		
		$posB     = $respT2[$stimType][$stimNum][$tframeA][$tframeB];
		
		if($posB    == 5) { $diffPosB = 3; } # easy
		elsif($posB == 1) { $diffPosB = 1; } # hard
		else              { $diffPosB = 2; } # med	 
		
		$posB_Avg = $respT2Avg[$stimType][$stimNum][$tframeA][$tframeB];
		if($posB_Avg    >= 5) { $diffPosB_Avg = 1; } # hard
		elsif($posB_Avg <= 2) { $diffPosB_Avg = 3; } # easy
		else                  { $diffPosB_Avg = 2; } # med
		
		$diffPos = $posA - $posB;
		if($diffPos    >=  4) { $descDiffPos = 3; }
		elsif($diffPos <= -4) { $descDiffPos = 1; }
		else                  { $descDiffPos = 2; }
		
		
		$T1_MIN  = $parts[10];
		$T1_MAX  = $parts[11];
		$T1_AVG  = $parts[12];
		$T1_STD  = $parts[13];
		$T1_MAXX = $parts[14];
		$T1_MAXY = $parts[15];
		$T1_AREA = $parts[16];
		
		print(SPSS "$parts[0],$posA,$posB,$posA_Avg,$posB_Avg,$diffPosA_Avg,$diffPosB_Avg,".
		           "$diffPosA,$diffPosB,$diffPos,$descDiffPos,".
		           "$parts[1],$parts[2],$parts[3],$parts[9],".
		           "$T1_MIN,$T1_MAX,$T1_AVG,$T1_STD,$T1_MAXX,$T1_MAXY,$T1_AREA,xxx,");
	    }
	    elsif($targ == 2) # T1
	    {
		$T2_MIN  = $parts[10];
		$T2_MAX  = $parts[11];
		$T2_AVG  = $parts[12];
		$T2_STD  = $parts[13];
		$T2_MAXX = $parts[14];
		$T2_MAXY = $parts[15];
		$T2_AREA = $parts[16];
		
		$DIFF_MIN  = $T1_MIN  - $T2_MIN;
		$DIFF_MAX  = $T1_MAX  - $T2_MAX;
		$DIFF_AVG  = $T1_AVG  - $T2_AVG;
		$DIFF_STD  = $T1_STD  - $T2_STD;
		$DIFF_MAXX = $T1_MAXX - $T2_MAXX;
		$DIFF_MAXY = $T1_MAXY - $T2_MAXY;
		$DIFF_AREA = $T1_AREA - $T2_AREA;
		
		$EIG_MIN  = sqrt($T1_MIN  * $T1_MIN  + $T2_MIN  * $T2_MIN);
		$EIG_MAX  = sqrt($T1_MAX  * $T1_MAX  + $T2_MAX  * $T2_MAX);
		$EIG_AVG  = sqrt($T1_AVG  * $T1_AVG  + $T2_AVG  * $T2_AVG);
		$EIG_STD  = sqrt($T1_STD  * $T1_STD  + $T2_STD  * $T2_STD);
		$EIG_MAXX = sqrt($T1_MAXX * $T1_MAXX + $T2_MAXX * $T2_MAXX);
		$EIG_MAXY = sqrt($T1_MAXY * $T1_MAXY + $T2_MAXY * $T2_MAXY);
		$EIG_AREA = sqrt($T1_AREA * $T1_AREA + $T2_AREA * $T2_AREA);
		
		$DIFF_DIST = sqrt(($DIFF_MAXX*$DIFF_MAXX) + ($DIFF_MAXY*$DIFF_MAXY)); 
		
		print(SPSS "$T2_MIN,$T2_MAX,$T2_AVG,$T2_STD,$T2_MAXX,$T2_MAXY,$T2_AREA,xxx,".
		           "$DIFF_MIN,$DIFF_MAX,$DIFF_AVG,$DIFF_STD,$DIFF_MAXX,$DIFF_MAXY,$DIFF_AREA,".
		           "$DIFF_DIST,xxx,".
		           "$EIG_MIN,$EIG_MAX,$EIG_AVG,$EIG_STD,$EIG_MAXX,$EIG_MAXY,$EIG_AREA\n");
	    }
	    else
	    {
		die "Unknown target type given as \"$targ\". It must be 1 or 2\n";
	    }
	}
	else
	{
	    #print("Skipping $parts[9] ne $CHANNEL\n");
	}
    }
    close(SURPRISE);
    close(SPSS);
}
