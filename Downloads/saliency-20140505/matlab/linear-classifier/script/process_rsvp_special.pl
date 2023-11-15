#!/usr/bin/perl 

# Manual version of process_rsvp.pl

$condition 	= "UHOLTXE_max"; 
$fdur 		= "50ms"; 
$redir 		= ">/dev/null 2>&1"; 

foreach $d (@ARGV) 
{
	@x = split(/\//, $d); 
	$fbase = pop(@x); 
	@base  = split(/\_/,$fbase);
	$newBase = "$base[0]\_$base[3]\_$base[4]\_$base[5]";

	# Clean each file before running since ezvision will append 
	$command = "rm -f $d/chan*txt"; 
	system($command); 

	# For each stim sequence call ezvision with this way 
	$command = "/lab/mundhenk/saliency/bin/ezvision ". 
	"--ior-type=None --agm-type=Std ". 
	"--surprise --vc-type=UHOLTXE ". 
	"--nouse-random --display-foa --ior-type=None ". 
	"--sm-type=Trivial --wta-type=None ". 
	"--in=$d/$newBase\_#.png --input-frames=0-MAX\@$fdur --output-frames=0-MAX\@$fdur --out=none ". 
	"--sv-type=Stats --sv-stats-fname=$d/stats.txt --save-channel-stats ". 
	"--save-channel-stats-name=$d/chan.txt ". 
	"--save-channel-stats-tag=$d_${fbase} --save-stats-per-channel ". 
	"--ag-type=Std ". 
	"--surprise-take-st-max --surprise-slfac=1.0 --surprise-ssfac=1.0 --surprise-neighsig=0.5 --surprise-locsig=3 ". 
	"$redir"; 

	# Make a record of each command called, lock the log while doing this 
	open(LOGFILE, ">>/lab/mundhenk/linear-classifier/log/process_rsvp.log.txt"); 
	flock LOGFILE, 2; 
	print(LOGFILE "RUNNING $command\n\n"); 
	close(LOGFILE); 
	flock LOGFILE, 8; 

	#Run the actual command 
	system($command); 
}
