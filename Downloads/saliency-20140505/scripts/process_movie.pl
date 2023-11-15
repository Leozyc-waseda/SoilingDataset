#!/usr/bin/perl

# Process a movie with the bottom-up visual attention model. Can
# handle .tbz movies (which should be a bzip2'd .tar archive
# containing all frames as frameXXXXXX.ppm; see movie2tbz.pl or the
# older avi2tbz.pl for how to create a .tbz from other movie
# files). Result is an mpeg, tbz or mov (need make_qt in the path) movie.
#
# USAGE: process_movie.pl [options] /path/to/movie1.tbz /path/to/movie2.tbz ...
#
# COMMAND LINE OPTIONS:  (must be given before file names)
#
# --tbz                save a .tbz movie as result rather than .mpg
# --m2v                save a .m2v movie as result (requires mpeg2encode)
# --mov                save a .mov movie as result (requires make_qt)
# --divx               save a .avi DivX movie as result (requires mencoder)
# --rawframes          don't encode a movie, just save the raw frames
# --create-orig        create a movie of the original unprocessed frames
# --parallel           use iLab Beowulf for mpeg encoding
# --idelay=x           load a new input frame every x milliseconds
# --odelay=x           write a new output frame every x milliseconds
# --smfac=x            use a saliency map rescaling factor of x
# --testmode           time-stamp result frames to fixed date before packing
# --vision-exec=x      specify full path to ezvision executable
# --longterm           use long-term temporary to avoid daily cleanups
# --save-textlog       save textlog output as input.txtlog
# --notmpdir           unpack movie in current dir rather than a temp dir
#
# Unrecognized options will be passed directly to vision.

# requirements: mpeg_encode, make_qt, ezvision, mpeg2encode, pnmtojpeg,
# mencoder in PATH

$parallel = 0; $orig = 0; if (-f "/etc/brshtab") { @nodes=`cat /etc/brshtab`; }
$idelay = 33.33333; $odelay = 33.33333; $longterm = 0; $textlog = 0;
$smfac = 200000;
$output_tbz = 0;
$output_mov = 0;
$testmode = 0;
$output_m2v = 0;
$output_divx = 0;
$output_rawframes = 0;
$usetmpdir = 1;

$cmd_untar = "tar jxf";
$cmd_vision = "ezvision";
$cmd_mpg = "mpeg_encode";
$cmd_m2v = "mpeg2encode";
$cmd_qt = "make_qt";
$cmd_tojpg = "pnmtojpeg";
$cmd_menc = "mencoder";
$defopts = " --test-mode --save-x-combo";
$options = ""; $didsomething = 0;

foreach $f (@ARGV) {
    chomp $f;
    # process command-line arguments:
    if ($f eq '--create-orig') {
        print STDERR "Will create movie from original frames\n";
        $orig = 1;
    } elsif ($f eq '--parallel') {
        print STDERR "Will use Beowulf for mpeg encoding\n";
        $parallel = 1;
    } elsif ($f eq '--tbz') {
        print STDERR "Will create .tbz output rather than .mpg\n";
        $output_tbz = 1;
    } elsif ($f eq '--testmode') {
        print STDERR "Turning testmode on.\n";
        $testmode = 1;
    } elsif ($f eq '--notmpdir') {
        print STDERR "Using current directory for temporary files\n";
        $usetmpdir = 0;
    } elsif ($f eq '--mov') {
        print STDERR "Will create .mov output rather than .mpg\n";
        check_exec($cmd_qt); $output_mov = 1;
    } elsif ($f eq '--m2v') {
        print STDERR "Will create .m2v output rather than .mpg\n";
        check_exec($cmd_mpg); $output_m2v = 1;
    } elsif ($f eq '--divx') {
        print STDERR "Will create .avi DivX output rather than .mpg\n";
        check_exec($cmd_tojpg);        check_exec($cmd_menc); $output_divx = 1;
    } elsif ($f eq '--rawframes') {
        print STDERR "Will save raw frames rather than encoding a movie\n";
        $output_rawframes = 1;
    } elsif (substr($f, 0, 9) eq '--idelay=') {
        @tmp = split(/=/, $f); $idelay = pop(@tmp);
        print STDERR "Loading new frame every $idelay ms\n";
    } elsif (substr($f, 0, 9) eq '--odelay=') {
        @tmp = split(/=/, $f); $odelay = pop(@tmp);
        print STDERR "Writing out new frame every $odelay ms\n";
    } elsif (substr($f, 0, 8) eq '--smfac=') {
        @tmp = split(/=/, $f); $smfac = pop(@tmp);
        print STDERR "Using saliency map factor of $smfac\n";
    } elsif (substr($f, 0, 14) eq '--vision-exec=') {
        @tmp = split(/=/, $f); $cmd_vision = pop(@tmp);
        print STDERR "Using vision executable at $cmd_vision\n";
    } elsif ($f eq '--longterm') {
        print STDERR "Using long-term temporary directory\n";
        $longterm = 1;
    } elsif ($f eq '--save-textlog') {
        print STDERR "Saving textlog\n";
        $textlog = 1;
    } elsif (substr($f, 0, 1) ne '/') { # anything not filename goes to vision
        $options .= ' ' . $f;
    } else {
        # check path and extract filename, path and extension:
        if (substr($f, 0, 1) ne '/') {
            die "Path to movies must be absolute -- ABORT.\n";
        }
        if (! -f $f) {
            die "No such file: $f -- ABORT.\n";
        }
        @tmp = split(/\//, $f); $fname = pop(@tmp); $base = join('/', @tmp);
        @tmp = split(/\./, $fname); $ext = pop(@tmp); $fbase = join('.', @tmp);
        $didsomething = 1;  # will never get here if path was not absolute

        # select temp directory:
        if ($usetmpdir)
        {
            if ($parallel) { $tdir = "$base/process_movie$$"; }
            elsif ($longterm) {
                if (-d "/home/tmp/u") { $tdir = "/home/tmp/u/process_movie$$"; }
                else { $tdir = "/tmp/process_movie$$"; }
            } else {
                if (-d "/home/tmp/1") { $tdir = "/home/tmp/1/process_movie$$"; }
                else { $tdir = "/tmp/process_movie$$"; }
            }
            if (! -d $tdir) {
                if (! mkdir($tdir)) {
                    print STDERR "couldn't make temp directory $tdir\n";
                    $tdir = "/tmp/process_movie$$";
                    mkdir($tdir) or die "couldn't make temp directory $tdir: $!\n";
                }
            }
            chdir($tdir) or die "couldn't cd to $tdir: $!\n";

            print STDERR "### Using temporary directory $tdir\n";
        } else {
            print STDERR "### Unpacking in current directory\n";
            $tdir = ".";
        }

        # extract movie depending on extension:
        print STDERR "### Unpacking $ext movie $base/$fname\n";
        if ($ext eq 'tbz') { system("$cmd_untar $f"); }
        else { die "Movie must be .tbz; see movie2tbz.pl\n"; }

        # get input frame range:
        $ff = `/bin/ls frame*.ppm|grep 'frame......\\.ppm'|head -1`; $ff=substr($ff,5,6);
        $lf = `/bin/ls frame*.ppm|grep 'frame......\\.ppm'|tail -1`; $lf=substr($lf,5,6);
        print STDERR "### Input frames: $ff - $lf\n";

        # get input file size:
        open F, "frame$ff.ppm" || die "Cannot read frame$ff.ppm\n";
        $go = 2; $siz = "";
        while($go) {
            $x = <F>; chomp $x;
            if (substr($x, 0, 1) ne '#')
            { $go --; if ($go == 0) { $x =~ s/\s+/x/g; $siz = $x} }
        }
        close F; if ($siz eq "") { die "Bogus file format for frame$ff.ppm\n";}
        print STDERR "### Frame size: $siz\n";

        # do we want to write the original movie out?
        if ($orig) {
            if ($output_divx && lc($ext) ne "avi")
            { encode_movie_divx("$base/$fbase.avi", $ff, $lf, $parallel,
                                $siz, $tdir, 'frame'); }
            elsif ($output_m2v && lc($ext) ne "m2v")
            { encode_movie_mpg2("$base/$fbase.m2v", $ff, $lf, $parallel,
                                $siz, $tdir, 'frame'); }
            elsif ($output_mov && lc($ext) ne "mov")
            { encode_movie_qt("$base/$fbase.mpg", $ff, $lf, $parallel,
                              $siz, $tdir, 'frame'); }
            elsif (lc($ext) ne "mpg")
            { encode_movie("$base/$fbase.mpg", $ff, $lf, $parallel,
                           $siz, $tdir, 'frame'); }
        }

        # check that the vision executable is in the path:
        check_exec($cmd_vision);

        # process the movie:
        if ($options) { $cmd_vision .= $options; $options = ''; $defopts = '';}
        if ($defopts) { $cmd_vision .= $defopts; $defopts = ''; }
        if ($textlog) { $cmd_vision .= " --textlog=log.txt"; }

        system("$cmd_vision --input-frames=$ff-$lf\@$idelay ".
               "--output-frames=0-1000000\@$odelay ".
               "--display-map-factor=$smfac ".
               "--nouse-fpe --in=raster:frame#.ppm --out=raster:");

        # copy the fixations if any:
        if (-s "frame.foa")
        { system("/bin/cp frame.foa $base/$fbase.foa"); }
        if (-s "log.txt")
        { system("/bin/cp log.txt $base/$fbase.txtlog"); }

        # copy a gmon.out file, if any:
        if (-s "gmon.out")
        { system("/bin/cp gmon.out $base/gmon.out"); }

        # get output frame range:
        $off = `/bin/ls T*.pnm|grep 'T......\\.pnm'|head -1`; $off = substr($off,1,6);
        $olf = `/bin/ls T*.pnm|grep 'T......\\.pnm'|tail -1`; $olf = substr($olf,1,6);
        print STDERR "### Output frames: $off - $olf\n";

        # encode the results:
        if ($output_rawframes) { # just save raw frames (no movie encoding)
            # rename the results TXXXXXX.pnm as
            # $base/rawframeXXXXXX.pnm, unless we are not using a temp
            # dir, in which case we just leave them in the current
            # directory:
            $ii = $off;
            while($ii <= $olf) {
                $orig = sprintf("T%06d.pnm", $ii);
                if ($usetmpdir)        { $dest = sprintf("%s/rawframe%06d.pnm", $base, $ii); }
                else { $dest = sprintf("rawframe%06d.pnm", $ii); }
                system("/bin/mv $orig $dest");
                print STDERR "moved $orig to $dest\n";
                $ii ++;
            }
        } elsif ($output_tbz) {  # do tbz encoding
            print STDERR "### Encoding .tbz result movie...\n";
            mkdir('tbz');
            # let's rename the results TXXXXXX.pnm as tbz/frameXXXXXX.pnm:
            $ii = $off;
            while($ii <= $olf) {
                rename(sprintf("T%06d.pnm", $ii),
                       sprintf("tbz/frame%06d.pnm", $ii));
                $ii ++;
            }
            chdir('tbz');
            # if in testmode, let's timestamp all files to fixed date
            # so that we don't get a different file due to different
            # dates:
            if ($testmode) {
                system("/bin/touch -d \"Jan 1 00:00:00 PST 2003\" . *");
                system("/bin/tar cf - --owner=root --group=root ".
                       "--portability --mode=0644 frame*.pnm | ".
                       "/usr/bin/bzip2 -9 >$base/${fbase}S.tbz");
            } else {
                system("/bin/tar cf - . | /usr/bin/bzip2 -9 ".
                       ">$base/${fbase}S.tbz");
            }
            chdir('..');
        } elsif ($output_mov) {  # to mov encoding
            encode_movie_qt("$base/${fbase}S.m2v", $off, $olf, $parallel,
                              $siz, $tdir, 'T');
        } elsif ($output_m2v) {  # do mpeg-2 encoding
            encode_movie_mpg2("$base/${fbase}S.m2v", $off, $olf, $parallel,
                              $siz, $tdir, 'T');
        } elsif ($output_divx) { # do DivX encoding
            encode_movie_divx("$base/${fbase}S.avi", $off, $olf, $parallel,
                              $siz, $tdir, 'T');
        } else {          # do mpeg-1 encoding
            encode_movie("$base/${fbase}S.mpg", $off, $olf, $parallel,
                         $siz, $tdir, 'T');
        }

        if ($usetmpdir)
        {
            # delete temporary files:
            chdir('..');
            system("/bin/rm -rf $tdir");
        }
    }
}
if ($didsomething == 0)
{ print STDERR "USAGE: process_movie.pl /path/to/movie.tbz\n"; }

######################################################################
sub encode_movie {  # name, first_frame, last_frame, parallel, size, tdir, fram
    check_exec($cmd_mpg);
    my $pname; $pname = "$_[5]/param.$$";
    open FF, ">$pname" || die "Cannot write $pname\n";
    print FF <<EOF;
PATTERN          IBBPBBPBBPBBPBB
OUTPUT           $_[0]
SIZE             $_[4]
INPUT_DIR        $_[5]
BASE_FILE_FORMAT PPM
GOP_SIZE         30
SLICES_PER_FRAME 1
PIXEL                 HALF
RANGE                 10
PSEARCH_ALG         LOGARITHMIC
BSEARCH_ALG         CROSS2
IQSCALE                 8
PQSCALE                 10
BQSCALE                 25
FORCE_ENCODE_LAST_FRAME 1
REFERENCE_FRAME         ORIGINAL
INPUT_CONVERT         \*
INPUT
$_[6]\*.pnm        [$_[1]-$_[2]]
END_INPUT
EOF
    if ($_[3]) { # use parallel encoder
        print FF "PARALLEL_TEST_FRAMES 3\nPARALLEL_TIME_CHUNKS 30\nPARALLEL\n";
        my $u; $u = `whoami`; chomp $u; my $n;
        foreach $n (@nodes) {
            chomp $n; print FF "$n $u /usr/bin/mpeg_encode\n";
        }
        print FF "END_PARALLEL\n"; fflush(FF);
        system("brsh ls $pname");
    }
    close FF; system('sync');
    print STDERR "### Encoding mpeg movie into $_[0]\n";
    system("$cmd_mpg $pname");
    unlink($pname);
}

######################################################################
sub encode_movie_qt { # name, first, last, parallel, size, tdir, fram
    check_exec($cmd_qt);
    system("$cmd_qt $_[1] $_[2] $_[6] $_[0]");
}

######################################################################
sub encode_movie_mpg2 {  # name, first, last, parallel, size, tdir, fram
    check_exec($cmd_m2v);
    my $pname; $pname = "$_[5]/param.$$";
    open FF, ">$pname" || die "Cannot write $pname\n";
    $nf = $_[2] - $_[1]; ($w, $h) = split(/x/, $_[4]);
    print FF <<EOF;
MPEG-2 Sequence, 30 frames/sec
$_[6]%06d /* name of source files */
-         /* name of reconstructed images ("-": dont store) */
-         /* name of intra quant matrix file     ("-": default matrix) */
-         /* name of non intra quant matrix file ("-": default matrix) */
-         /* name of statistics file ("-": stdout ) */
2         /* input picture file format: 0=*.Y,*.U,*.V, 1=*.yuv, 2=*.ppm */
$nf       /* number of frames */
$_[1]     /* number of first frame */
00:00:00:00 /* timecode of first frame */
8         /* N (nb of frames in GOP) */
2         /* M (I/P frame distance) */
0         /* ISO/IEC 11172-2 stream */
0         /* 0:frame pictures, 1:field pictures */
$w        /* horizontal_size */
$h        /* vertical_size */
1         /* aspect_ratio_information 8=CCIR601 625 line, 9=CCIR601 525 line */
5         /* frame_rate_code 1=23.976, 2=24, 3=25, 4=29.97, 5=30 frames/sec. */
6000000.0 /* bit_rate (bits/s) */
256       /* vbv_buffer_size (in multiples of 16 kbit) */
0         /* low_delay  */
1         /* constrained_parameters_flag */
4         /* Profile ID: Simple=5, Main=4, SNR=3, Spatial=2, High=1 */
6         /* Level ID:   Low=10, Main=8, High 1440=6, High=4        */
1         /* progressive_sequence */
1         /* chroma_format: 1=4:2:0, 2=4:2:2, 3=4:4:4 */
0         /* video_format: 0=comp., 1=PAL, 2=NTSC, 3=SECAM, 4=MAC, 5=unspec. */
5         /* color_primaries */
5         /* transfer_characteristics */
5         /* matrix_coefficients */
$w        /* display_horizontal_size */
$h        /* display_vertical_size */
2         /* intra_dc_precision 0: 8 bit, 1: 9 bit, 2: 10 bit, 3: 11 bit */
0         /* top_field_first */
1 1 1     /* frame_pred_frame_dct (I P B) */
0 0 0     /* concealment_motion_vectors (I P B) */
0 0 0     /* q_scale_type  (I P B) */
0 0 0     /* intra_vlc_format (I P B)*/
0 0 0     /* alternate_scan (I P B) */
0         /* repeat_first_field */
1         /* progressive_frame */
0         /* P distance between complete intra slice refresh */
0         /* rate control: r (reaction parameter) */
0         /* rate control: avg_act (initial average activity) */
0         /* rate control: Xi (initial I frame global complexity measure) */
0         /* rate control: Xp (initial P frame global complexity measure) */
0         /* rate control: Xb (initial B frame global complexity measure) */
0         /* rate control: d0i (initial I frame virtual buffer fullness) */
0         /* rate control: d0p (initial P frame virtual buffer fullness) */
0         /* rate control: d0b (initial B frame virtual buffer fullness) */
2 2 11 11 /* P:  forw_hor_f_code forw_vert_f_code search_width/height */
1 1 3  3  /* B1: forw_hor_f_code forw_vert_f_code search_width/height */
1 1 7  7  /* B1: back_hor_f_code back_vert_f_code search_width/height */
1 1 7  7  /* B2: forw_hor_f_code forw_vert_f_code search_width/height */
1 1 3  3  /* B2: back_hor_f_code back_vert_f_code search_width/height */
EOF
    close FF; system('sync');
    print STDERR "### Encoding mpeg-2 movie into $_[0]\n";
    system("$cmd_m2v $pname $_[0] > /dev/null");
    unlink($pname);
}
######################################################################
sub encode_movie_divx {  # name, first, last, parallel, size, tdir, fram
    check_exec($cmd_menc); check_exec($cmd_tojpg);
    my $ofps = 1000.0 / $odelay;

    # convert output frames to JPEG (converting to PNG does not work
    # with the current mencoder if color depth is not 24-bit, and both
    # 'pnmtopng' and 'convert' will use non-24bit colormapped color
    # depths if your images contain few colors):
    print STDERR "### Converting frames to JPEG...\n"; my $ii = $_[1];
    while($ii <= $_[2]) {
        system(sprintf("$cmd_tojpg --quality=100 $_[6]%06d.pnm>$_[6]%06d.jpg",
                       $ii, $ii)); $ii ++;
    }
    # do the encoding proper:
    print STDERR "### Encoding DivX movie into $_[0]\n";
    system("$cmd_menc -o $_[0] -noskip -ovc lavc -lavcopts vcodec=mpeg4 ".
           "-mf on:type=jpeg:fps=$ofps -info name=\"$_[0]\":artist=".
           "\"process_movie.pl\":copyright=\"iLab - University of Southern ".
           "California\" \"$_[6]*.jpg\"");
    print STDERR "### Cleaning up intermediary files...\n";
    $ii = $_[1]; while($ii <= $_[2]) { unlink(sprintf("$_[6]%06d.jpg",$ii++));}
}

######################################################################
sub check_exec { # exec_name
    my @stuff = split(/\s+/, $_[0]);  # discard command arguments if any
    my $where = `/usr/bin/which $stuff[0]`;
    chomp $where; if (length($where) < 3)
    { die "Cannot find $stuff[0] -- ABORT.\n"; }
}
