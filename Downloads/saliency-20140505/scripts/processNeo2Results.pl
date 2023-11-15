#!/usr/bin/perl


BEGIN { push(@INC, "./scripts"); }
use strict;
use List::Util qw[min max];

srand;                      # not needed for 5.004 and later

my $resultsDir = shift;
my $testsuiteDir = shift;

#Write the svn version
system("echo Results:saliency `svnversion .` >> $resultsDir/SvnVersion");
system("echo Results:neo2 `svnversion src/Apps/neo2` >> $resultsDir/SvnVersion");


opendir(DIR, $resultsDir) || die "Can not open $resultsDir $!\n";
foreach my $module (readdir(DIR))
{
	next if ($module =~ /^\./);
	next if (!(-d "$resultsDir/$module"));
	
	next if ($module eq "Random");

	my $moduleDir = "$resultsDir/$module";
	my $testRootDir = "$resultsDir/$module/Testing";


	opendir(TRAINDIR, $testRootDir) || die "Can not open $testRootDir $!\n";
	foreach my $trainDir (readdir(TRAINDIR))
	{
		next if ($trainDir =~ /^\./);
	  my $testDir = "$testRootDir/$trainDir";

		my $combinedResultsFile = "$testDir/results.txt";
		my $combinedClassesFile = "$testDir/classes.txt";
    #clear the combined results file
		open(FILE, ">$combinedResultsFile");close(FILE);
		open(FILE, ">$combinedClassesFile");close(FILE);


		opendir(TESTDIR, $testDir) || die "Can not open $testDir $!\n";
		foreach my $testSceneDir (readdir(TESTDIR))
		{
			next if ($testSceneDir =~ /^\./);
			my $resultsDir = "$testDir/$testSceneDir";
			next if (!(-d $resultsDir));
			print "Generating results for $resultsDir\n";
			&generateResults($resultsDir, $combinedResultsFile, $combinedClassesFile);
		}
		closedir(TESTDIR);
		&generateResults($testDir); 
	}
	closedir(TRAINDIR);
}
closedir(DIR);


sub generateResults()
{
  my($dir,$resultsFile, $classesFile) = @_;

  my %CLASS_INFO;
  my %CLASS_NAME;
  my %CLASS_BK;

  if (defined $classesFile)
  {
    if (!open(CLSFILE, ">>$classesFile"))
    {
      print "Can not open $classesFile ($!)\n";
      return;
    }
  }

#read the class information
  if (!open(FILE, "$dir/classes.txt")) 
  {
     print "Can not open $dir/classes.txt ($!)\n";
     return;
  }
  while(<FILE>)
  {
    print CLSFILE if (defined $classesFile);
    chomp;
    my ($classId, $isBackground, $numInstances, $name) = split(/\s+/, $_, 4);

    $CLASS_BK{$classId} = $isBackground;
    $CLASS_NAME{$classId} = $name;

    if (defined $CLASS_INFO{$classId})
    {
      $CLASS_INFO{$classId} +=$numInstances;
    } else {
      $CLASS_INFO{$classId} =$numInstances;
    }


  }
  close(FILE);
  close(CLSFILE) if (defined $classesFile);



#read the results information
  if (defined $resultsFile)
  {
    if (!open(RESULTSFILE, ">>$resultsFile"))
    {
      print "Can not open $resultsFile ($!)\n";
      return;
    }
  }

  if (!open(FILE, "$dir/results.txt"))
  {
    print "Can not open $dir/results.txt ($!)\n";
    return;
  }
  my $totalTime = 0;
  my $numFrames = 0;
  my $lastFrame = -1;
  my $totalPolygons = 0;

  my %classScores;
  my @trueIds;
  my @maxScores;
  my @maxIds;


  while(<FILE>)
  {
    print RESULTSFILE if (defined $resultsFile);
    chomp();
    my ($frameNum, $time, $classID, $objConf, $tlx, $tly, $brx, $bry, $id_scores) = split(/\s+/, $_, 9);
    $totalPolygons++;

    #Calculate the number of frames
    if ($lastFrame ne $frameNum)
    {
      $lastFrame = $frameNum;
      $numFrames++;
      $totalTime += $time;
    }

    push(@trueIds, $classID);

		my %idScores = &getIdScores($id_scores);
		#my %idScores = &getIdScoresRandom($id_scores);
		#check if we have the current class in the array,
		#otherwise it has a score of 0
		$idScores{$classID} = 0 if (!defined $idScores{$classID});

    #Get the max Ids and scores for the combined ROC curve
    my ($maxId, $maxScore) = &findMax(%idScores);
    push(@maxIds, $maxId);
    push(@maxScores, $maxScore);

    #Push the scores into an array of classes
		foreach my $objId (keys %idScores)
		{
      if (!defined $classScores{$objId})
      {
        my @tmp;
        $classScores{$objId} = \@tmp;
      } 
      my $arrayPtr = $classScores{$objId};
      push(@$arrayPtr, $idScores{$objId});
    }

  }
  close(FILE);
  close(RESULTSFILE) if (defined $resultsFile);

  #Senity check
  if ($totalPolygons <= 0)
  {
     print "Invalid results files\n";
     return;
  }


#Output statistics

  foreach my $classId (keys %CLASS_INFO)
  {
		my $scoresPtr = $classScores{$classId};

		if (!open(RESULTS, ">$dir/$classId.roc"))
		{
			print "Can not open $dir/$classId.roc for writing ($!)\n";
			return; 
		}

		print RESULTS $CLASS_NAME{$classId}, "\n";
		print RESULTS $CLASS_INFO{$classId}, "\n";

		my ($threshPtr, $fpPtr, $tpPtr, $totalTP, $totalFP) = &getROC($scoresPtr, \@trueIds, $classId);
		print RESULTS "$totalFP $totalTP\n";

		my @thresh = @$threshPtr;
		my @fp = @$fpPtr;
		my @tp = @$tpPtr;

		for(my $i=0; $i<$#fp+1; $i++)
		{
			print RESULTS $thresh[$i], " ",  sprintf("%0.2f", $fp[$i]), " ", sprintf("%0.2f", $tp[$i]), "\n";
		}
		close(RESULTS);
	}

#Output a combined ROC curve
	if (!open(RESULTS, ">$dir/all.roc"))
	{
		print "Can not open $dir/all.roc for writing ($!)\n";
		return; 
	}
	print RESULTS "AllClasses\n";
	print RESULTS "-1\n";

	my ($threshPtr, $fpPtr, $tpPtr, $totalTP, $totalFP) = &getAllROC(\@maxScores, \@maxIds, \@trueIds);
	print RESULTS "$totalFP $totalTP\n";

	my @thresh = @$threshPtr;
	my @fp = @$fpPtr;
	my @tp = @$tpPtr;

	for(my $i=0; $i<$#fp+1; $i++)
	{
		print RESULTS $thresh[$i], " ",  sprintf("%0.2f", $fp[$i]), " ", sprintf("%0.2f", $tp[$i]), "\n";
	}
	close(RESULTS);

#Output global results

	if (!open(RESULTS, ">$dir/global.txt"))
	{
		print "Can not open $dir/global.txt for writing ($!)\n";
		return;
	}


	print RESULTS "TotalPolygons $totalPolygons\n";
	print RESULTS "NumFrames $numFrames\n";
	print RESULTS "TotalTime $totalTime\n";
	print RESULTS "AverageFrameRate " , sprintf("%0.2f", $numFrames/$totalTime), "\n";
	print RESULTS "AveragePolygonsPerFrame " , sprintf("%0.2f", $totalPolygons/$numFrames), "\n";
	print RESULTS "AveragePolygonPerSec " , sprintf("%0.2f", $totalPolygons/$totalTime), "\n";
	close(RESULTS);

#Output a confusion matrix
	if (!open(CONFU, ">$dir/confusion.txt"))
	{
		print "Can not open $dir/confusion.txt for writing ($!)\n";
		return;
	}

	for(my $idx=0; $idx<$#maxIds+1; $idx++)
	{
		print CONFU "$trueIds[$idx] $maxIds[$idx]\n";
	}
	close(CONFU);


}

sub getIdScores()
{
	my ($id_scoresStr) = @_;

#Get the class IDS and Scores
	my @scoreArray = split(/\s+/, $id_scoresStr);
	my @objIds = ();
	my @objScores = ();

	my %idScores;
	for(my $i=0; $i < $#scoreArray/2; $i+=1)
	{
		$idScores{$scoreArray[$i]} = $scoreArray[$i+($#scoreArray/2)+1];
	}

	return %idScores;

}

sub getIdScoresRandom()
{
	my ($id_scoresStr) = @_;

#Get the class IDS and Scores
	my @scoreArray = split(/\s+/, $id_scoresStr);

	my %idScores;
	for(my $i=0; $i < $#scoreArray/2; $i+=1)
	{
		$idScores{$scoreArray[$i]} = 0;
	}

  #Pick a random element from the list and set it to 1
  my @ids = keys %idScores;
  my $index   = rand @ids;
	$idScores{$ids[$index]} = 1;

	return %idScores;

}


sub getROC()
{
	my($scores, $trueIds, $classId) = @_;

	my @scoresArray = @$scores;
	my @trueIdsArray = @$trueIds;

#Compute the score for only the class id that we wanted
	my @results;
	my $totalTP = 0;
	my $totalFP = 0;

	for(my $i=0; $i<$#scoresArray+1; $i++)
	{
		my $match = 0;
		if ($trueIdsArray[$i] eq $classId)
		{
			$totalTP++;
			$match = 1; 
		} else {
			$totalFP++;
			$match = 0; 
		}
		my @data = ($scoresArray[$i], $match);
		push(@results, \@data);
	}

#Sweep the threshold

	my @fp;
	my @tp;
	my @thresh;
  my $minThresh = min(@scoresArray);
  my $maxThresh = max(@scoresArray);
  my $range = $maxThresh-$minThresh;

	#for(my $th=$maxThresh; $th >= 0; $th -= $range/100)
	for(my $thIdx=100; $thIdx >= 0; $thIdx -= 1)
	{
		my $fpSum = 0;
		my $tpSum = 0;
		my $th = $minThresh + ($range*$thIdx/100);


		foreach my $id (@results)
		{
			if ($id->[0] >= $th)
			{
				$tpSum++ if ($id->[1] == 1); 
				$fpSum++ if ($id->[1] == 0);
			}
		}
		push(@tp, $tpSum);
		push(@fp, $fpSum);
		push(@thresh, $th);
	}


##Efficent search per Pascal matlab file
#my @thresh = sort{$b->[0] <=> $a->[0]} @results;
#	my @fp;
#	my @tp;
#	my $fpSum = 0;
#	my $tpSum = 0;
#	foreach my $th (@thresh)
#	{
#		$tpSum++ if ($th->[1] == 1); 
#		$fpSum++ if ($th->[1] == 0);
#		push(@tp, $tpSum);
#		push(@fp, $fpSum);
#
#	}


	return (\@thresh, \@fp, \@tp, $totalTP, $totalFP);
}

sub getAllROC()
{
	my($scores, $classIds, $trueIds) = @_;

	my @scoresArray = @$scores;
	my @classIdsArray = @$classIds;
	my @trueIdsArray = @$trueIds;

#Compute the score for only the class id that we wanted
	my @results;
	my $totalTP = 0;
	my $totalFP = 0;

	for(my $i=0; $i<$#scoresArray+1; $i++)
	{
		my $match = 0;
		if ($trueIdsArray[$i] eq $classIdsArray[$i])
		{
			$match = 1; 
		} else {
			$match = 0; 
		}
		$totalTP++;
		$totalFP++;
		my @data = ($scoresArray[$i], $match);
		push(@results, \@data);
	}

#Sweep the threshold

  #Sweep through the threshold

	my @fp;
	my @tp;
	my @thresh;
  my $minThresh = min(@scoresArray);
  my $maxThresh = max(@scoresArray);
  my $range = $maxThresh-$minThresh;
	$range = 0.1 if ($range <= 0);

	for(my $thIdx=100; $thIdx >= 0; $thIdx -= 1)
	{
		my $fpSum = 0;
		my $tpSum = 0;
		my $th = $minThresh + ($range*$thIdx/100);
		foreach my $id (@results)
		{
			if ($id->[0] >= $th)
			{
				$tpSum++ if ($id->[1] == 1); 
				$fpSum++ if ($id->[1] == 0);
			}
		}
		push(@tp, $tpSum);
		push(@fp, $fpSum);
		push(@thresh, $th);
	}


##Efficent search per Pascal matlab file
#my @thresh = sort{$b->[0] <=> $a->[0]} @results;
#	my $fpSum = 0;
#	my $tpSum = 0;
#	foreach my $th (@thresh)
#	{
#		$tpSum++ if ($th->[1] == 1); 
#		$fpSum++ if ($th->[1] == 0);
#		push(@tp, $tpSum);
#		push(@fp, $fpSum);
#
#	}
	return (\@thresh, \@fp, \@tp, $totalTP, $totalFP);
}



sub findMax()
{
	my(%idScores) = @_;

	my $maxId = -1;
	my $maxScore = -1;
	foreach my $key (keys %idScores)
	{
		if ($idScores{$key} > $maxScore)
		{
			$maxScore = $idScores{$key};
			$maxId = $key;
		}
	}

	return ($maxId, $maxScore);
}

