$command = "ls -1";

@files = qx"$command";

foreach(@files)
{
    chomp();
    $file = $_;
    if($file =~ "\.tbz")
    {
	print("Skipping $file\n");
    }
    else
    {
	$command = "ls -1 $file";
	@subfiles = qx"$command";
	foreach(@subfiles)
	{
	    chomp();
	    $subfile = $_;
	    if($subfile =~ "new-")
	    {
		$command = "rm -Rf $file\/$subfile";
		system("$command");
	    }
	}
    }
}
	
