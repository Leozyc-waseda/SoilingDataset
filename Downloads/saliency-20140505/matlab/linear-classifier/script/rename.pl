@DIRS = qx"ls -1";

foreach (@DIRS)
{
    chomp();
    if($_ =~ "stim")
    {
	$d = $_;
	@parts = split(/\_/);
	$stim  = $parts[1];
	@parts = split(/m/,$parts[0]);
	$frame = $parts[1];
	if($stim > 99)
	{
	    print("Stim $stim\n");
	    @FILES = qx"ls -1 $d/stim_*.png";
	    foreach (@FILES)
	    {
		chomp();
		$file = $_;
		@parts = split(/\_/);
		$newname = "$parts[0]\_$stim\_$parts[2]";
		$command = "mv $file $d/$newname";
		print("COMMAND: $command\n");
		system($command);
	    }
	}
    }
}
