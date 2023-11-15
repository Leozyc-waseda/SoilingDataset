$LEADIN_FRAMES  = 10; # use to ADD EXTRA lead in frames
$LEADOUT_FRAMES = 2;  # use to ADD EXTRA lead out frames

#$FILELIST = "filelist.txt";
$FILELIST   = "Partial-Repete-AB.filelist.txt";
#If using EXT3 we are limited on how many links we can make compaired with XFS
$USING_EXT3 = 1;



open(FILELIST, "$FILELIST");
while(<FILELIST>)
{
    chomp();
    $line = $_;
    if(substr($line,0,1) eq "#")
    {
	@dir = split(/\#/,$line);
	print("Making $dir[1]\n"); 
	$command = "mkdir $dir[1]";
	system($command);
	$frameCount = 0;
	
	# add leadin images
	if($USING_EXT3)
	{
	    $frameForm = sprintf("%06d", $frameCount);
	    $fixFrame = "$dir[1]/$dir[1]\_$frameForm.png";
	    $command = "cp ../fixate.png $fixFrame\n";
	    system($command);
	    $frameCount++;
	    for($n = 1; $n < $LEADIN_FRAMES; $n++)
	    {
		$frameForm = sprintf("%06d", $frameCount);
		$command = "ln $fixFrame $dir[1]/$dir[1]\_$frameForm.png\n";
		system($command);
		$frameCount++;
	    } 
	}
	else
	{
	    for($n = 0; $n < $LEADIN_FRAMES; $n++)
	    {
		$frameForm = sprintf("%06d", $frameCount);
		$command = "ln ../fixate.png $dir[1]/$dir[1]\_$frameForm.png\n";
		system($command);
		$frameCount++;
	    }
	}
    }
    elsif(substr($line,0,1) eq "")
    {
	# add leadout images

	for($n = 0; $n < $LEADOUT_FRAMES; $n++)
	{
	    $frameForm = sprintf("%06d", $frameCount);
	    $command = "ln $fixFrame $dir[1]/$dir[1]\_$frameForm.png\n";
	    system($command);
	    $frameCount++;
	}	
    }
    else
    {
	$frameForm = sprintf("%06d", $frameCount);
	@file = split(/ /,$line);
	$command = "ln $file[0] $dir[1]/$dir[1]\_$frameForm.png\n";
	system($command);
	$frameCount++;
    }
}
