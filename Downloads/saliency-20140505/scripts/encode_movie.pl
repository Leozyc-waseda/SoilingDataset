#!/usr/bin/perl

# Encode a list of images into a movie. Frames must be in PPM format
# and must be named <basename>XXXXXX.ppm where XXXXXX is a six-digit
# frame number (frame numbers must be contiguous). If and only if no
# file matching <basename>XXXXXX.ppm is found then a second try will
# be attempted using a .pnm extension instead of .ppm (this can also
# be manually set by specifying --pnm on the command line).
#
# USAGE: encode_movie.pl [OPTS] <basename> <output.mpg>
#
# COMMAND LINE OPTIONS:  (must be given before file names)
#
# --mpg           save a .mpg MPEG-1 movie (requires mpeg_encode)
# --mpghq         save a high-quality .mpg MPEG-1 movie (requires mpeg_encode)
# --parallel      use iLab Beowulf for MPEG-1 encoding
# --nodes="n01 ..." use specified nodes for parallel encoding
# --tbz           save a .tbz movie
# --m2v           save a .m2v MPEG-2 movie (requires mpeg2encode)
# --mov           save a .mov Quicktime movie (requires make_qt)
# --divx          save a .avi DivX movie (requires mencoder)
# --yuv           assume YUV420P inputs rather than PPM (only for mpg/mpghq)
# --pnm           assume PPM files but with .pnm extension
# --frames=x-y    force using frames number x to y (both inclusive)
#
# if no option is specified, --mpg will be assumed.
#
##########################################################################
## The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   ##
## by the University of Southern California (USC) and the iLab at USC.  ##
## See http://iLab.usc.edu for information about this project.          ##
##########################################################################
## Major portions of the iLab Neuromorphic Vision Toolkit are protected ##
## under the U.S. patent ``Computation of Intrinsic Perceptual Saliency ##
## in Visual Environments, and Applications'' by Christof Koch and      ##
## Laurent Itti, California Institute of Technology, 2001 (patent       ##
## pending; application number 09/912,225 filed July 23, 2001; see      ##
## http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     ##
##########################################################################
## This file is part of the iLab Neuromorphic Vision C++ Toolkit.       ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is free software; you can   ##
## redistribute it and/or modify it under the terms of the GNU General  ##
## Public License as published by the Free Software Foundation; either  ##
## version 2 of the License, or (at your option) any later version.     ##
##                                                                      ##
## The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  ##
## that it will be useful, but WITHOUT ANY WARRANTY; without even the   ##
## implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      ##
## PURPOSE.  See the GNU General Public License for more details.       ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   ##
## to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   ##
## Boston, MA 02111-1307 USA.                                           ##
##########################################################################
##
## Primary maintainer for this file: Laurent Itti <itti@usc.edu>
## $Id: encode_movie.pl 10337 2008-10-13 17:24:09Z itti $
##

$cmd_mpg = "mpeg_encode";
$cmd_m2v = "mpeg2encode";
$cmd_qt = "make_qt";
$cmd_menc = "mencoder";
$cmd_tojpg = "pnmtojpeg";
$usage = "USAGE: encode_movie.pl [OPTS] <basename> <output.mpg>\n";

if ($#ARGV < 1) { die $usage; }
$base = ''; $out = ''; $ie = 'ppm'; $ff = -1, $lf = -1;

foreach $f (@ARGV) {
    chomp $f;
    # process command-line arguments:
    if ($f eq '--parallel') {
        print STDERR "Will use Beowulf for mpeg encoding\n";
        $parallel = 1;
    } elsif ($f eq '--tbz') {
        print STDERR "Will create TBZ output\n";
        $enc = 'tbz';
    } elsif (substr($f, 0, 7) eq '--nodes') {
        my @gogo = split(/=/, $f); my $n = $gogo[1];
        print STDERR "Will use nodes: $n\n";
        @nodes = split(/\s+/, $n);
    } elsif (substr($f, 0, 8) eq '--frames') {
        ($ff, $lf) = split(/\-/, substr($f, 9, 20));
        $ff = sprintf('%06d', $ff);
        $lf = sprintf('%06d', $lf);
    } elsif ($f eq '--mpg') {
        print STDERR "Will create MPEG-1 output\n";
        check_exec($cmd_mpg); $enc = 'mpg';
    } elsif ($f eq '--mpghq') {
        print STDERR "Will create high-quality MPEG-1 output\n";
        check_exec($cmd_mpg); $enc = 'mpghq';
    } elsif ($f eq '--mov') {
        print STDERR "Will create Quicktime output\n";
        check_exec($cmd_qt); $enc = 'qt';
    } elsif ($f eq '--m2v') {
        print STDERR "Will create MPEG-2 output\n";
        check_exec($cmd_m2v); $enc = 'm2v';
    } elsif ($f eq '--divx') {
        print STDERR "Will create DivX output\n";
        check_exec($cmd_tojpg);        check_exec($cmd_menc); $enc = 'divx';
    } elsif ($f eq '--yuv') {
        print STDERR "Assuming YUV inputs rather than PPM\n";
        $ie = 'yuv';
    } elsif ($f eq '--pnm') {
        print STDERR "Assuming PPM inputs with .pnm file extension\n";
        $ie = 'pnm';
    } elsif ($base eq '') {
        $base = $f;
    } elsif ($out eq '') {
        $out = $f;
    } else { die $usage; }
}

if ($out eq '' || $base eq '') { die $usage; }
if ($enc eq '') { $enc = 'mpg'; }
if ($parallel && $#nodes == -1) {
    $n = `/bin/cat /etc/brshtab | xargs`; chomp($n);
    @nodes = split(/\s+/, $n);
}
# get input frame range:
if ($ff == -1 && $lf == -1) {
    $ff = `/bin/ls $base??????.$ie|head -1`;
    $ff=substr($ff, length($base), 6);
    $lf = `/bin/ls $base??????.$ie|tail -1`;
    $lf=substr($lf, length($base), 6);
    if ($ie eq 'ppm' && length($ff) < 1) { # no frames? try again with .pnm
        $ie = 'pnm';
        $ff = `/bin/ls $base??????.$ie|head -1`;
        $ff=substr($ff, length($base), 6);
        $lf = `/bin/ls $base??????.$ie|tail -1`;
        $lf=substr($lf, length($base), 6);
    }
}

print STDERR "### Input frames: $ff - $lf. Output: $out\n";

# get input file size:
if ($ie eq 'ppm' || $ie eq 'pnm')
{
    open F, "$base$ff.$ie" || die "Cannot read $base$ff.$ie\n";
    $go = 2; $siz = "";
    while($go) {
        $x = <F>; chomp $x;
        if (substr($x, 0, 1) ne '#')
        { $go --; if ($go == 0) { $x =~ s/\s+/x/g; $siz = $x} }
    }
    close F; if ($siz eq "") { die "Bogus file format for $base$ff.$ie\n";}
    print STDERR "### Frame size: $siz\n";
}
else
{
    print STDERR "### Assuming YUV frame size of 640x480\n";
    $siz = '640x480';
}

@x = split(/\//, $base); $fname = pop(@x); $path = join('/', @x);
if ($path eq '') { $path = '.'; }

# encode the movie:
if ($enc eq 'tbz') {  # do tbz encoding
    die "BOGUS IMPLEMENTATION";
    $tdir = "$path/tbz$$";
    print STDERR "### Encoding .tbz result movie...\n";
    mkdir($tdir);
    # let's copy the results as tbz/frameXXXXXX.ppm:
    $ii = $ff;
    while($ii <= $lf) {
        system(sprintf("/bin/cp $path/${fname}%06d.$ie $tdir/frame%06d.$ie",
                       $ii, $ii));
        $ii ++;
    }
    chdir($tdir);
    # if in testmode, let's timestamp all files to fixed date
    # so that we don't get a different file due to different
    # dates:
    if ($testmode) {
        system("/bin/touch -d \"Jan 1 00:00:00 PST 2003\" . *");
        system("/bin/tar cf - --owner=root --group=root ".
               "--portability --mode=0644 frame*.$ie | ".
               "/usr/bin/bzip2 -9 > $out");
    } else {
        system("/bin/tar cf - . | /usr/bin/bzip2 -9 > $out");
    }
    chdir('..');
    system("/bin/rm -rf tbz$$");
} elsif ($enc eq 'mov') {  # to mov encoding
    encode_movie_qt($out, $ff, $lf, $base);
} elsif ($enc eq 'm2v') {  # do mpeg-2 encoding
    encode_movie_mpg2($out, $ff, $lf, $siz, $base);
} elsif ($enc eq 'divx') { # do DivX encoding
    encode_movie_divx($out, $ff, $lf, $fname, $ie);
} elsif ($enc eq 'mpg') {  # do mpeg-1 encoding
    encode_movie($out, $ff, $lf, $parallel, $siz, $path, $fname, $ie);
} elsif ($enc eq 'mpghq') {  # do high-quality mpeg-1 encoding
    encode_moviehq($out, $ff, $lf, $parallel, $siz, $path, $fname, $ie);
} else { die $usage; }

######################################################################
sub encode_movie {  # name, first, last, parallel, size, path, fname, inext
    check_exec($cmd_mpg); my $pname;
    if ($_[3]) { $pname = "/lab/tmpi1/1/param.$$"; }
    else { $pname = "/tmp/param.$$"; }
    open FF, ">$pname" || die "Cannot write $pname\n";
    my $fmt;
    if ($_[7] eq 'ppm' || $_[7] eq 'pnm') { $fmt = 'PPM'; }
    else { $fmt = 'YUV'; }
    print FF <<EOF;
PATTERN          IBBPBBPBBPBBPBB
OUTPUT           $_[0]
SIZE             $_[4]
INPUT_DIR        $_[5]
BASE_FILE_FORMAT $fmt
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
$_[6]\*.$_[7]        [$_[1]-$_[2]]
END_INPUT
EOF

    if ($_[3]) { # use parallel encoder
        print FF "PARALLEL_TEST_FRAMES 3\nPARALLEL_TIME_CHUNKS 30\nPARALLEL\n";
        my $u; $u = `whoami`; chomp $u; my $n;
        foreach $n (@nodes) {
            chomp $n; print FF "$n $u /usr/bin/mpeg_encode\n";
            print "$n: ";
            system("rsh $n /bin/ls -l /usr/bin/mpeg_encode");
        }
        print FF "END_PARALLEL\n";
        #system("brsh ls $pname");
    }
    close FF; system('sync');
    print STDERR "### Encoding mpeg movie into $_[0]\n";
    system("$cmd_mpg $pname");
    unlink($pname);
}

######################################################################
sub encode_moviehq {  # name, first, last, parallel, size, path, fname
    check_exec($cmd_mpg); my $pname;
    if ($_[3]) { $pname = "/lab/tmpi1/1/param.$$"; }
    else { $pname = "/tmp/param.$$"; }
    open FF, ">$pname" || die "Cannot write $pname\n";
    my $fmt;
    if ($_[7] eq 'ppm' || $_[7] eq 'pnm') { $fmt = 'PPM'; }
    else { $fmt = 'YUV'; }
    print FF <<EOF;
PATTERN          IBBPBBPBBPBBPBB
OUTPUT           $_[0]
SIZE             $_[4]
INPUT_DIR        $_[5]
BASE_FILE_FORMAT $fmt
GOP_SIZE         30
SLICES_PER_FRAME 1
PIXEL                 HALF
RANGE                 10
PSEARCH_ALG         LOGARITHMIC
BSEARCH_ALG         CROSS2
IQSCALE                 1
PQSCALE                 1
BQSCALE                 1
FORCE_ENCODE_LAST_FRAME 1
REFERENCE_FRAME         ORIGINAL
INPUT_CONVERT         \*
INPUT
$_[6]\*.$_[7]        [$_[1]-$_[2]]
END_INPUT
EOF

    if ($_[3]) { # use parallel encoder
        print FF "PARALLEL_TEST_FRAMES 3\nPARALLEL_TIME_CHUNKS 30\nPARALLEL\n";
        my $u; $u = `whoami`; chomp $u; my $n;
        foreach $n (@nodes) {
            chomp $n; print FF "$n $u /usr/bin/mpeg_encode\n";
            print "$n: ";
            system("rsh $n /bin/ls -l /usr/bin/mpeg_encode");
        }
        print FF "END_PARALLEL\n";
        #system("brsh ls $pname");
    }
    close FF; system('sync');
    print STDERR "### Encoding mpeg movie into $_[0]\n";
    system("$cmd_mpg $pname");
    unlink($pname);
}

######################################################################
sub encode_movie_qt { # name, first, last, base
    check_exec($cmd_qt);
    system("$cmd_qt $_[1] $_[2] $_[3] $_[0]");
}

######################################################################
sub encode_movie_mpg2 {  # name, first, last, siz, base
    check_exec($cmd_m2v); my $pname; $pname = "/tmp/param.$$";
    open FF, ">$pname" || die "Cannot write $pname\n";
    $nf = $_[2] - $_[1]; ($w, $h) = split(/x/, $_[3]);
    print FF <<EOF;
MPEG-2 Sequence, 30 frames/sec
$_[4]%06d /* name of source files */
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
sub encode_movie_divx {  # name, first, last, base, ext
    check_exec($cmd_menc); check_exec($cmd_tojpg);
    my $ofps = 1000.0 / $odelay;

    # convert output frames to JPEG (converting to PNG does not work
    # with the current mencoder if color depth is not 24-bit, and both
    # 'pnmtopng' and 'convert' will use non-24bit colormapped color
    # depths if your images contain few colors):
    print STDERR "### Converting frames to JPEG...\n"; my $ii = $_[1];
    while($ii <= $_[2]) {
        system(sprintf("$cmd_tojpg --quality=100 $_[3]%06d.$_[4] > ".
                       "$_[3]%06d.jpg", $ii, $ii)); $ii ++;
    }
    # do the encoding proper:
    print STDERR "### Encoding DivX movie into $_[0]\n";
    system("$cmd_menc -o $_[0] -noskip -ovc lavc -lavcopts vcodec=mpeg4 ".
           "-mf on:type=jpeg:fps=$ofps -info name=\"$_[0]\":artist=".
           "\"process_movie.pl\":copyright=\"iLab - University of Southern ".
           "California\" \"$_[3]*.jpg\"");
    print STDERR "### Cleaning up intermediary files...\n";
    $ii = $_[1]; while($ii <= $_[2]) { unlink(sprintf("$_[3]%06d.jpg",$ii++));}
}

######################################################################
sub check_exec { # exec_name
    my @stuff = split(/\s+/, $_[0]);  # discard command arguments if any
    my $where = `/usr/bin/which $stuff[0]`;
    chomp $where; if (length($where) < 3)
    { die "Cannot find $stuff[0] -- ABORT.\n"; }
}
