# unpack all tbz or tgz files in the current directory

#$command = "ls -1 *.tbz";
$command = "ls -1 *.tgz";

@files = qx"$command";

foreach $fl (@files)
{
    chomp($fl);
#    $command = "bzip2 -dc $fl | tar -xf -";
    $command = "gzip -dc $fl | tar -xf -";
    print("COMMAND: $command\n");
    system($command);
}
