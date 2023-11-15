#!/usr/bin/perl

# USAGE: process_rsvp.pl <stim1> ... <stimN>
# each stim is a directory with a bunch of PNG files in it

$condition = "JointGG_UCIO_basic";

$fdur = '50ms';  # duration of each frame
$redir = '>/dev/null 2>&1';

foreach $d (@ARGV) {
    @x = split(/\//, $d); $fbase = pop(@x);
    
    #$newdir = "/lab/tmpib/30/rsvp/${fbase}";
    #$command = "mkdir $newdir";

    #open(LOGFILE, ">>/lab/mundhenk/linear-classifier/log/process_rsvp.log.txt");
    #flock LOGFILE, 2;
    #print(LOGFILE "MKDIR $command\n\n");
    #close(LOGFILE);
    #flock LOGFILE, 8;

    #system($command);

    if($condition eq "UCIO_legacy")
    {
	$command = "/lab/nerd-cam/saliency.code.archive/saliency.jul.2006/bin/ezvision ".
	"--movie --ior-type=None ".
	"-X --surprise --vc-type=UCIO ".
	"--nouse-random --display-foa --ior-type=None ".
	"--sm-type=Trivial --wta-type=None ".
	"--in=$d/${fbase}_#.png --input-frames=0-MAX\@$fdur ".
	"--output-frames=0-MAX\@$fdur ".
	"--out=png:$d/stim --save-salmap ".
	"--salmap-factor=1e12 --agm-factor=1e12 ".
	"--sv-stats-fname=$d/stats.txt ".
	"--save-channel-stats ".
	"--save-channel-stats-name=$d/chan.txt ".
	"--save-channel-stats-tag=$d_${fbase} ". 
	"$redir";
    }
    elsif($condition eq "UCIO_old")
    {
	$command = "/lab/mundhenk/saliency/bin/ezvision ".
	"--movie --ior-type=None --agm-type=Std ".
	"--surprise --vc-type=UC:0.8333I:0.4167O:1.6667F:0.4167M:1.6667 ".
	"--display-map=SM --display-map-factor=50000 --vcx-normsurp=0 ".
	"--nouse-random --display-foa --ior-type=None ".
	"--sm-type=Trivial --wta-type=None ".
	"--in=$d/$d_${fbase}_#.png --input-frames=0-MAX\@$fdur ".
	"--out=mpeg:$d/stim- --output-frames=0-MAX\@$fdur ".
        "--output-frames=0-MAX\@$fdur ".
	"--out=png:$d/stim --save-salmap ".
	"--sv-type=Stats --sv-stats-fname=$d/stats.txt --save-channel-stats ".
	"--save-channel-stats-name=$d/chan.txt ".
	"--save-channel-stats-tag=$d_${fbase} ".
	"$redir";  
    }
    elsif($condition eq "NATHAN_UCIO_basic")
    {
	$command = "/lab/mundhenk/saliency/bin/ezvision ".
	"--movie --ior-type=None --agm-type=Std ".
	"--surprise --vc-type=UCIO ".
	"--nouse-random --display-foa --ior-type=None ".
	"--sm-type=Trivial --wta-type=None ".
	"--in=$d/$d_${fbase}_#.png --input-frames=0-MAX\@$fdur ".
	"--out=mpeg:$d/stim- --output-frames=0-MAX\@$fdur ".
        "--output-frames=0-MAX\@$fdur ".
	"--out=png:$d/stim --save-salmap ".
	"--sv-type=Stats --sv-stats-fname=$d/stats.txt --save-channel-stats ".
	"--save-channel-stats-name=$d/chan.txt ".
	"--save-channel-stats-tag=$d_${fbase} --surprise-type=Nathan ".
	"$redir";  
    }  
    elsif($condition eq "JointGG_UCIO_basic")
    {
	$command = "/lab/mundhenk/saliency/bin/ezvision ".
	"--movie --ior-type=None --agm-type=Std ".
	"--surprise --vc-type=UCIO ".
	"--nouse-random --display-foa --ior-type=None ".
	"--sm-type=Trivial --wta-type=None ".
	"--in=$d/$d_${fbase}_#.png --input-frames=0-MAX\@$fdur ".
	"--out=mpeg:$d/stim- --output-frames=0-MAX\@$fdur ".
        "--output-frames=0-MAX\@$fdur ".
	"--out=png:$d/stim --save-salmap ".
	"--sv-type=Stats --sv-stats-fname=$d/stats.txt --save-channel-stats ".
	"--save-channel-stats-name=$d/chan.txt ".
	"--save-channel-stats-tag=$d_${fbase} --surprise-type=JointGG ".
	"$redir";  
    }
    elsif($condition eq "UHIO_basic")
    {
	$command = "/lab/mundhenk/saliency/bin/ezvision ".
	"--movie --ior-type=None --agm-type=Std ".
	"--surprise --vc-type=UHIO ".
	"--nouse-random --display-foa --ior-type=None ".
	"--sm-type=Trivial --wta-type=None ".
	"--in=$d/$d_${fbase}_#.png --input-frames=0-MAX\@$fdur ".
	"--out=mpeg:$d/stim- --output-frames=0-MAX\@$fdur ".
        "--output-frames=0-MAX\@$fdur ".
	"--out=png:$d/stim --save-salmap ".
	"--sv-type=Stats --sv-stats-fname=$d/stats.txt --save-channel-stats ".
	"--save-channel-stats-name=$d/chan.txt ".
	"--save-channel-stats-tag=$d_${fbase} ".
	"$redir";  
    }
    elsif($condition eq "UCIO_basic")
    {
	$command = "/lab/mundhenk/saliency/bin/ezvision ".
	"--movie --ior-type=None --agm-type=Std ".
	"--surprise --vc-type=UCIO ".
	"--nouse-random --display-foa --ior-type=None ".
	"--sm-type=Trivial --wta-type=None ".
	"--in=$d/$d_${fbase}_#.png --input-frames=0-MAX\@$fdur ".
	"--out=mpeg:$d/stim- --output-frames=0-MAX\@$fdur ".
        "--output-frames=0-MAX\@$fdur ".
	"--out=png:$d/stim --save-salmap ".
	"--sv-type=Stats --sv-stats-fname=$d/stats.txt --save-channel-stats ".
	"--save-channel-stats-name=$d/chan.txt ".
	"--save-channel-stats-tag=$d_${fbase} ".
	"$redir";  
    }
    else
    {
	die("ERROR UNKNOWN CONDITION \"$condition\"\n");
    } 

    open(LOGFILE, ">>/lab/mundhenk/linear-classifier/log/process_rsvp.log.txt");
    flock LOGFILE, 2;
    print(LOGFILE "RUNNING $command\n\n");
    close(LOGFILE);
    flock LOGFILE, 8;
    system($command);

}



##### variance:
#    system("/lab/tmpi1/u/rsvp/ezvision ".
#	   "--movie --ior-type=None ".
#	   "--nouse-random --display-foa --ior-type=None ".
#	   "--sm-type=Trivial --wta-type=None ".
#"--vc-type=Variance --maxnorm-type=Ignore --nouse-older-version --chanout-max=0 -X ".
#	   "--in=$d/${fbase}_#.png --input-frames=0-MAX\@$fdur ".
#	   "--out=mpeg:$d --output-frames=0-MAX\@$fdur ".
#"--salmap-factor=1e12 --agm-factor=1e12 ".
#	   "--sv-stats-fname=${d}.txt ".
#	   "$redir");

##### surprise:
#    system("/lab/tmpi1/u/rsvp/ezvision ".
#	   "--movie --ior-type=None ".
#	   "-K --surprise ".
#	   "--nouse-random --display-foa --ior-type=None ".
#	   "--sm-type=Trivial --wta-type=None ".
#	   "--in=$d/${fbase}_#.png --input-frames=0-MAX\@$fdur ".
#	   "--out=mpeg:$d --output-frames=0-MAX\@$fdur ".
#	   "--salmap-factor=3e10 --agm-factor=3e10 ".
#	   "--sv-stats-fname=${d}.txt ".
#	   "$redir");

##### saliency:
#    system("/lab/tmpi1/u/rsvp/ezvision ".
#	   "--movie --ior-type=None ".
#	   "--nouse-random --display-foa --ior-type=None ".
#	   "--sm-type=Trivial --wta-type=None ".
#"--vc-type=CIOFM --maxnorm-type=FancyOne --nouse-older-version --chanout-max=0 --direction-sqrt --gabor-intens=20.0 -K ".
#	   "--in=$d/${fbase}_#.png --input-frames=0-MAX\@$fdur ".
#	   "--out=mpeg:$d --output-frames=0-MAX\@$fdur ".
#	   "--salmap-factor=3e10 --agm-factor=3e10 ".
#"--salmap-factor=1e11 --agm-factor=1e11 ".
#	   "--sv-stats-fname=${d}.txt ".
#	   "$redir");
