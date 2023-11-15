#!/usr/bin/perl
use POSIX;
#a stupid little script to just grab all of the onset times 
#in MSECS for a particular movie and aparticular subject.  

#input: <test.psy> <prefix if any>

#load our file
open F, "<$ARGV[0]";
@fdat = <F>;
close(F);

$subnum = $ARGV[1];

#step through the file getting all of our data.  
my @data;
foreach $line (@fdat){
    chomp($line); 
    ($t, $o) = split(/\s+/, $line, 2);
 
    if ($o =~ /Playing movie/) {
 
	local @tmp;
        
	#OK we are either at the beginning or the end of a sequence
        #so if there are values, lets write them out, 
	if (length(@data) > 0) {
	    open FOUT, ">$subnum$movie.mtf";	
	    print(FOUT @data);
	    close(FOUT); 
	}

        #reset some vars
	@data = "";
	#get the movie name
	@tmp = split(/\s+/,$o);
	$movie = pop(@tmp);
	$movie = pop(@tmp);
	@tmp = split(/\//,$movie);
	$movie = pop(@tmp);
	
    }
    elsif ($o =~ /displayYUVoverlay - frame 0/) { #first frame 
	$tref = $t;
	push(@data,0);
    }
    elsif ($o =~  /displayYUVoverlay - frame/) { #any other frame
	$diff = (utim($t) - utim($tref)) / 1000;
	push(@data,"$diff\n");
	#print "$diff\n";
    }
}
    




sub utim { # time
    my @x = split(/[:\.]/, $_[0]);
    return $x[0] * 3.6e9 + $x[1] * 6.0e7 + $x[2] * 1.0e6 +
	$x[3] * 1.0e3 + $x[4];
}
