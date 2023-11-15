$MASK_SET_LIST = "/lab/tmpib/u/maskSequence/Short_Easy_list_AZ.txt";
$HARD_SET_LIST = "/home/storage/surpriseReduction/allPresent_S3.hard.txt";
$MASK_DIR      = "/lab/tmpib/u/maskSequence/";
$HARD_DIR      = "/lab/tmpib/u/hardSequence/";
$OUTPUT_DIR    = "/lab/tmpib/u/maskSequenceHard/";
$OUTPUT_LIST   = "$OUTPUT_DIR/subject_list.txt";
$HARD_AFFIX    = "_hard_w";

open(MASK_SET, "$MASK_SET_LIST") or die "Cannot open \'$MASK_SET_LIST\' for reading\n";
open(HARD_SET, "$HARD_SET_LIST") or die "Cannot open \'$HARD_SET_LIST\' for reading\n";

while(<HARD_SET>)
{
    chomp();
    $line = $_;
    @parts = split(/\t/);
    
    # is 1 if is w type
    if($parts[12] == 1)
    {
	# store this line in a hash map
	$hardLines{"$parts[0]"} = $line;
    }
}

open(OUTPUT,">$OUTPUT_LIST") or die "Cannot open \'$OUTPUT_LIST\' for writing\n";

while(<MASK_SET>)
{
    chomp();
    $mask_line  = $_;
    @mask_parts = split(/\t/);
    @mask_specA = split(/\_/,$mask_parts[0]);
    @mask_specB = split(/m/,$mask_specA[0]);

    # print name and stuff like : stim06_003	06	003
    print(OUTPUT "$mask_parts[0]\t$mask_specB[1]\t$mask_specA[1]\t");
    for($i = 1; $i < 9; $i++)
    {
	print(OUTPUT "$mask_parts[$i]\t");
    }
    print(OUTPUT "$mask_parts[9]\t0\tMASK\n");

    $hard_line  = $hardLines{"$mask_parts[0]"};
    @hard_parts = split(/\t/,$hard_line);
    print(OUTPUT "$hard_line");
    print(OUTPUT "HARD\n");

    # copy "easy" mask files
    $command = "cp -R $MASK_DIR\/$mask_parts[0] $OUTPUT_DIR\/$mask_parts[0]";
    print("COPY \"$command\"\n");
    system("$command");

    # copy corresponding harder files
    $command = "cp -R $HARD_DIR\/$mask_parts[0]$HARD_AFFIX $OUTPUT_DIR\/$mask_parts[0]$HARD_AFFIX";
    print("COPY \"$command\"\n");
    system("$command");
}
