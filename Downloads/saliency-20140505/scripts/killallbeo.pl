for($i = 1; $i <= 16; $i++)
{
    if($i < 10)
    {
        $systemCall = "rsh n0$i killall $ARGV[0] $ARGV[1]";
        print("$systemCall\n");
        system("$systemCall");
    }
    else
    {
        $systemCall = "rsh n$i killall $ARGV[0] $ARGV[1]";
        print("$systemCall\n");
        system("$systemCall");
    }
}
