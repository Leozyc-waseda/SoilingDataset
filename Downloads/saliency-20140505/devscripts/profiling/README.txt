(1) Do runtests.sh with whatever options you want to test.

    This will generate a bunch of log.txt and gmon.out files

(2) Move all of the log.txt and gmon.out files to some dedicated
    directory; let's call it run1.

(3) Run runprofs.sh with the data directory name to compute all of the
    profiling information:

    ./devscripts/profiling/runprofs.sh run1/

    This generates a whole series of files, one set for each gmon.out
    file. Included in the set are:

    run1/*.txt    profile summary from gprof

    run1/*.pl1
    run1/*.pl     perl Data::Dumper form of profile information

    run1/*.dot    dot file with a profile call graph

    run1/*.gif
    run1/*.png
    run1/*.ps     various image formats with the call graph

    run1/*.map
    run1/*.html   html and image-map illustrating the call graph

(4) Run scrapelogs.pl with the data directory name to extract
    time/space complexity stats from the log files and profile files:

    ./devscripts/profiling/scrapelogs.pl run1/

    This generates run1/stats.txt, whose data can then be plotted in
    matlab using devscripts/profiling/plotstats.m:

    >> addpath devscripts/profiling
    >> plotstats('run1/stats.txt', 'run1/stats.eps');

    This will save the generated figure as 'run1/stats.eps'.

(5) Convert the stats eps file into a nicely antialiased png with
    devscripts/render_postscript:

    ./devscripts/render_postscript \
        run1/stats.eps w1600 4 | pnmtopng > run1/stats.png

(6) Run makeindex.sh to make an html index page that links to all of
    the aforementioned graphs and plots:

    ./devscripts/profiling/makeindex.sh run1

    which will generate run1/index.html. The title of the generated
    html page will be taken as the contents, if any, of a file named
    run1/TITLE; similarly, the body of the generated html page will
    include the contents, if any, of a file named run1/DESCRIPTION.
