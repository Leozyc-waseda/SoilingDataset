$command  = "ls -1";
print("$command\n");
@filelist = qx"$command";
foreach $f (@filelist)
{
    chomp($f);
    $command = "tar -czvf $f\.tgz $f";
    print("COMMAND $command\n");
    system("$command");
}
