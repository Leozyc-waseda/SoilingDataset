open(SUBJECTDATA,    "AY_forNathanWITHCONF.txt") or die "Cant open\n";
# We use a different output file for matlab since it gets
# confused by strings when parsing input files. SPSS however does not
# so we make one output file for each
open(SPSS_OUTFILE,    ">AY_SPSS_SubjectParsed.txt");
open(MATLAB_OUTFILE,  ">AY_MATLAB_SubjectParsed.txt");
open(MASTER_COND_FILE, ">AY_MATLAB_MasterCond.txt");

$m_conditions = 1;

# other condition information is in the directory names
# so we need to get them
$command = "ls -1";
@files = qx"$command";
$count = 1;
foreach $f (@files)
{    
    print("$f\n");
    if($f =~ "stim")
    {
	@parts = split(/\_/,$f);               # split stimXX_###
	$num   = sprintf("%03d",$parts[1]);    # get ###
	@spec  = split(/m/,$parts[0]);         # split stimXX
	$cond  = $spec[1];                     # get XX
	$DIR_SPEC{$num} = "$cond";             # store condition in hash table
	if(!(defined $DIR_SPEC_COND{$cond}))   # store the condition in a string free form
	{
	    print(MASTER_COND_FILE "$cond\n");
	    $DIR_SPEC_COND{$cond} = $count;
	    $count++;
	}
	$DIR_SPEC_NUM{$num} = $DIR_SPEC_COND{$cond}; # map condition name to number
    }
}

#for each stim (row in the input file)
while(<SUBJECTDATA>)
{
    chomp();
    my @line = split(/\,/);
    my $count = 0;
    my $col_count  = 0;
    my $bw_count   = 0;
    my $both_count = 0;
    my $none_count = 0;

    resetValsT1();
    resetValsT2();

    my $targ      = 0; 
    my $condition = 0;
    my @mcode     = (0,0);

    foreach $l (@line)
    {
	if($count != 0)
	{
	    if($count == 1)
	    {
		$targ = $l;
		print(MATLAB_OUTFILE "$l,"); 
	    }
	    if($count == 2)
	    {
		$condition = $l;
		# Use a hash table to track the different conditions and whether 
		# we have used them yet.
		if(!(defined $CodeChk{$l}) || ($CodeChk{$l} == 0))
		{
		    $CodeChk{$l} = $m_conditions;
		    $m_conditions++;
		}
		print(MATLAB_OUTFILE "$CodeChk{$l},"); 
	    }
	    elsif($count%3 == 0)
	    {
		#print("checking $l\n");
		$col = 0;
		$bw  = 0;
		if($l =~ "col")
		{
		    $col_count++;
		    $col = 1;
		    $mcode[0] = 1; 
		}
		elsif($l =~ "b/w")
		{
		    $bw_count++;
		    $bw = 1;
		    $mcode[1] = 1; 
		}
		elsif($l =~ "2")
		{
		    $col_count++;
		    $bw_count++;
		    $both_count++;
		    $col = 1;
		    $bw  = 1;
		    $mcode[0] = 1; 
		    $mcode[1] = 1; 
		}
		else
		{
		    $none_count++;
		}
		print(MATLAB_OUTFILE "$mcode[0],$mcode[1],");
	    }
	    elsif(($count-1)%3 == 0)
	    {
		crossCountT1($col,$bw,$l);
		print(MATLAB_OUTFILE "$l,");
	    } 
	    elsif(($count-2)%3 == 0)
	    {
		crossCountT2($col,$bw,$l);
		print(MATLAB_OUTFILE "$l,");
	    } 	
	}
	else
	{
	    my $stim_num      = sprintf("%03d",$l);       # format stim number
	    my $stim_cond     = $DIR_SPEC{$stim_num};     # get condition name
	    my $stim_cond_num = $DIR_SPEC_NUM{$stim_num}; # get the conditions id number
	    print(SPSS_OUTFILE   "$stim_cond,"); 
	    print(MATLAB_OUTFILE "$stim_cond_num,");  
	    print(MATLAB_OUTFILE "$l,");  
	}
	print(SPSS_OUTFILE "$l,");
	$count++;
    }

    print(SPSS_OUTFILE   " $col_count, $bw_count, $both_count, $none_count ");
    print(MATLAB_OUTFILE "$col_count,$bw_count,$both_count,$none_count");

    printValsT1();
    printValsT2();
    
    print(SPSS_OUTFILE   "\n");
    print(MATLAB_OUTFILE "\n");
}
close(SPSS_OUTFILE);

######################################################################
sub getMean    
{
    my $val   = $_[0];
    my $count = $_[1];
    if($count != 0) {$mean = $val/$count;}
    else            {$mean = "";}
    return $mean;
}

######################################################################
sub crossCountT1
{
    my $col  = $_[0];
    my $bw   = $_[1];
    my $val  = $_[2];

    print("$col, $bw, $val\n");

    if(($col) && ($bw))     # spotted both targets
    {
	$T1[1][1] += $val;
	$T1_COUNT[1][1]++;
    }
    elsif(($col) && !($bw)) # spotted color target, not bw target
    {
	$T1[1][0] += $val;
	$T1_COUNT[1][0]++;
    }
    elsif(!($col) && ($bw)) # spotted bw target, not color target
    {
	$T1[0][1] += $val;
	$T1_COUNT[0][1]++;
    }
    else                    # spottet neither
    {
	$T1[0][0] += $val;
	$T1_COUNT[0][0]++;
    }
}

######################################################################
sub crossCountT2
{
    my $col  = $_[0];
    my $bw   = $_[1];
    my $val  = $_[2];

    if(($col) && ($bw))     # spotted both targets
    {
	$T2[1][1] += $val;
	$T2_COUNT[1][1]++;
    }
    elsif(($col) && !($bw)) # spotted color target, not bw target
    {
	$T2[1][0] += $val;
	$T2_COUNT[1][0]++;
    }
    elsif(!($col) && ($bw)) # spotted bw target, not color target
    {
	$T2[0][1] += $val;
	$T2_COUNT[0][1]++;
    }
    else                    # spottet neither
    {
	$T2[0][0] += $val;
	$T2_COUNT[0][0]++;
    }
}

######################################################################
sub printValsT1
{
    for(my $x = 0 ;$x < 2; $x++)
    {
	for(my $y = 0 ;$y < 2; $y++)
	{	
	    my $mean = getMean($T1[$x][$y],$T1_COUNT[$x][$y]);
	    print(SPSS_OUTFILE   ",$mean");
	    print(MATLAB_OUTFILE ",$mean");
	}
    }
}

######################################################################
sub printValsT2
{
    for(my $x = 0 ;$x < 2; $x++)
    {
	for(my $y = 0 ;$y < 2; $y++)
	{	
	    my $mean = getMean($T2[$x][$y],$T2_COUNT[$x][$y]);
	    print(SPSS_OUTFILE   ",$mean");
	    print(MATLAB_OUTFILE ",$mean");
	}
    }
}

######################################################################
sub resetValsT1
{
    for(my $x = 0 ;$x < 2; $x++)
    {
	for(my $y = 0 ;$y < 2; $y++)
	{	
	    $T1[$x][$y]       = 0;
	    $T1_COUNT[$x][$y] = 0; 
	}
    } 
}

######################################################################
sub resetValsT2
{
    for(my $x = 0 ;$x < 2; $x++)
    {
	for(my $y = 0 ;$y < 2; $y++)
	{	
	    $T2[$x][$y]       = 0;
	    $T2_COUNT[$x][$y] = 0; 
	}
    } 
}
