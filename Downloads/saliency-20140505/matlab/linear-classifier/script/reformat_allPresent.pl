
useCert(4,"allResults.txt","allPresent.txt");

sub useCert 
{
    $subjects = $_[0];
    $infile   = $_[1];
    $outfile  = $_[2];

    open(ALLPRES,    "$infile")   or die "Cannot open input \"$infile\"\n";
    open(ALLPRESNEW, ">$outfile") or die "Cannot open output \"$outfile\"\n";
    while(<ALLPRES>)
    {
	chomp();
	@line     = split(/\,/);
	@name1    = split(/_/,$line[0]);
	if($name1[0] =~ "noTarg")         
	{
	    # skip no target
	}
	else
	{
	    @name2     = split(/m/,$name1[0]);
	    $tframe    = $name2[1];
	    $sample    = $name1[1];
	    $name      = $line[0];
	    $posScore  = 0;
	    $negScore  = 0;
	    $posCert   = 0;
	    $negCert   = 0;
	    $totalCert = 0;

	    print(ALLPRESNEW "$name\t$tframe\t$sample\t");
	    for($i = 0; $i < $subjects; $i++)
	    {
		$subjScore = $line[1 + $i];	 
		if(($subjScore <= 3) && ($subjScore != 0))  # Spotted Target
		{ 
		    $posScore++;                            # Count
		    $posCert += $subjScore;                 # Sum for averaging
		}  
		else                                                               # Did not spot the target
		{
		    if($subjScore == 0) { $newSubjScore = 1; $subjScore = 6; }     # Zero is actually 10
		    if($subjScore == 9) { $newSubjScore = 2; $subjScore = 5; }     # recode
		    if($subjScore == 8) { $newSubjScore = 3; $subjScore = 4; }     # recode 
		    $negScore++;                                                   # Count
		    $negCert += $newSubjScore;                                     # Sum for averaging
		}
		
		$totalCert += $subjScore;

		print(ALLPRESNEW "$line[1 + $i]\t");
	    }

	    $meanTotalCert = $totalCert / $subjects;   # mean score for all certanty
	    if($posScore > 0)
	    {
		$meanPosCert   = $posCert   / $posScore;   # mean score over detected targets
	    }
	    else
	    {
		$meanPosCert   = -1;
	    }

	    if($negScore > 0)
	    {
		$meanNegCert   = $negCert   / $negScore;   # mean score over missed targets
	    }
	    else
	    {
		$meanNegCert   = -1;
	    }

	    print(ALLPRESNEW "$posScore\t$negScore\t$meanTotalCert\t$meanPosCert\t$meanNegCert\n");
	}
    }
}

sub yesNoOnly
{
    
    open(ALLPRES,    "allPresent_S3.txt")      or die "Cannot open \"allPresent_S3.txt\"\n";
    open(ALLPRESNEW, ">allPresent_S3.new.txt") or die "Cannot open \"allPresent_S3.new.txt\"\n";
    while(<ALLPRES>)
    {
	chomp();
	@line  = split(/\,/);
	@name1 = split(/_/,$line[0]);
	$name  = "$name1[0]$name1[1]\_$name1[2]";
	$score = 0;
	print(ALLPRESNEW "$name\t$name1[1]\t$name1[2]\t");
	for($i = 0; $i < 8; $i++)
	{
	    $score += $line[1 + $i];
	    print(ALLPRESNEW "$line[1 + $i]\t");
	}
	print(ALLPRESNEW "$score\n");
    }
}
    
