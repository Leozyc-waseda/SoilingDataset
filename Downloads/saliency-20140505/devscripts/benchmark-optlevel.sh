#!/bin/sh

# $Id: benchmark-optlevel.sh 4489 2005-06-13 05:46:46Z rjpeters $

# Here are the results of running this script with different
# optimization levels, on a 2.8 GHz Xeon (ilab9.usc.edu),
# kernel-2.6.11, gcc-3.4.3, 2005-Jun-07

# I did a fresh svn checkout of saliency into multiple directories,
# one for each optimization level:

# cd /scratch
# svn checkout svn://ilab.usc.edu/trunk/saliency saliency-O0
# svn checkout svn://ilab.usc.edu/trunk/saliency saliency-O1
# svn checkout svn://ilab.usc.edu/trunk/saliency saliency-O2
# svn checkout svn://ilab.usc.edu/trunk/saliency saliency-O3
# svn checkout svn://ilab.usc.edu/trunk/saliency saliency-O4

# and then ran this script in each directory:

# cd /scratch/saliency-O0; ./devscripts/benchmark-optlevel.sh -O0
# cd /scratch/saliency-O1; ./devscripts/benchmark-optlevel.sh -O1
# cd /scratch/saliency-O2; ./devscripts/benchmark-optlevel.sh -O2
# cd /scratch/saliency-O3; ./devscripts/benchmark-optlevel.sh -O3
# cd /scratch/saliency-O4; ./devscripts/benchmark-optlevel.sh -O4

# and finally here are the results:
# (from running "cat /scratch/saliency-O*/optlevel-stats.log")

# [-O0] make all:      	594.93user 59.46system 10:55.37elapsed  99%CPU
# [-O0] make test:     	 41.78user  4.29system  0:46.01elapsed 100%CPU
# [-O0] make testlong: 	896.23user 41.37system 15:36.22elapsed 100%CPU
# [-O0] make test:     SUMMARY: TESTS FAILED!!! (13 of 193)
# [-O0] make testlong: SUMMARY: TESTS FAILED!!! (11 of 12)
# [-O1] make all:      	755.66user 64.58system 13:40.90elapsed  99%CPU
# [-O1] make test:     	 25.35user  5.69system  0:31.04elapsed 100%CPU
# [-O1] make testlong: 	315.97user 39.45system  5:53.59elapsed 100%CPU
# [-O1] make test:     SUMMARY: TESTS FAILED!!! (4 of 193)
# [-O1] make testlong: SUMMARY: TESTS FAILED!!! (4 of 12)
# [-O2] make all:      1041.30user 71.27system 18:34.45elapsed  99%CPU
# [-O2] make test:       22.61user  5.43system  0:28.15elapsed  99%CPU
# [-O2] make testlong:  279.89user 44.69system  5:22.65elapsed 100%CPU
# [-O2] make test:     SUMMARY: TESTS FAILED!!! (2 of 193)
# [-O2] make testlong: SUMMARY: TESTS FAILED!!! (4 of 12)
# [-O3] make all:      1079.06user 74.45system 19:17.11elapsed  99%CPU
# [-O3] make test:       21.39user  5.30system  0:26.65elapsed 100%CPU
# [-O3] make testlong:  281.95user 45.57system  5:26.14elapsed 100%CPU
# [-O3] make test:     SUMMARY: ALL TESTS PASSED (193 of 193)
# [-O3] make testlong: SUMMARY: ALL TESTS PASSED (12 of 12)
# [-O4] make all:      1087.62user 73.93system 19:23.46elapsed  99%CPU
# [-O4] make test:       21.43user  5.76system  0:27.33elapsed  99%CPU
# [-O4] make testlong:  275.91user 44.00system  5:19.85elapsed 100%CPU
# [-O4] make test:     SUMMARY: ALL TESTS PASSED (193 of 193)
# [-O4] make testlong: SUMMARY: ALL TESTS PASSED (12 of 12)

# summary:

# * Compile time increases steadily from -O0 to -O1 (+27%) and from -O1
#   to -O2 (+37%), then increases a bit more (+4%) from -O2 to -O3
#
# * Test suite time also decreases from -O0 to -O1 (-40%) and from -O1
#   to -O2 (-11%), and then decreases a bit more (-5%) from -O2 to -O3
#   (after repeating this last comparison a few times, the difference
#   might be as large as -10%).
#
# * Testlong time seems to be steady between -O2 and -O3.
#
# * No major differences between -O3 and -O4.

optlevel=$1

if test x$optlevel = x; then
    echo "must specify an optimization level"
    exit 1
fi

dir=`dirname $0`
case $dir in
    /*)
	# ok, we already have an absolute path
	;;
    *)
	# need to make the path absolute by prepending 'pwd'
	dir=$PWD/`dirname $0`
	;;
esac

toplev=${dir}/..

cd $toplev

if test -e $toplev/config.log; then
    echo "config.log already exists; skipping 'configure'"
else
    ./configure --enable-shlibs --enable-optimization=$optlevel
fi

if test -e $toplev/make.log; then
    echo "make.log already exists; skipping 'make all'"
else
    /usr/bin/time make all 2>&1 | tee $toplev/make.log
    chmod -w $toplev/make.log
fi

cd $toplev/tests

if test -e $toplev/test.log; then
    echo "test.log already exists; skipping 'make test'"
else
    ./run_test_suite.tcl 2>&1 | tee $toplev/test.log
    chmod -w $toplev/test.log
fi

if test -e $toplev/test-benchmark.log; then
    echo "test-benchmark.log already exists; skipping test benchmark"
else
    /usr/bin/time ./run_test_suite.tcl --nocomparison 2>&1 | tee $toplev/test-benchmark.log
    chmod -w $toplev/test-benchmark.log
fi

if test -e $toplev/testlong.log; then
    echo "testlong.log already exists; skipping 'make testlong'"
else
    ./run_testlong_suite.tcl 2>&1 | tee $toplev/testlong.log
    chmod -w $toplev/testlong.log
fi

if test -e $toplev/testlong-benchmark.log; then
    echo "testlong-benchmark already exists; skipping testlong benchmark"
else
    /usr/bin/time ./run_testlong_suite.tcl --nocomparison 2>&1 | tee $toplev/testlong-benchmark.log
    chmod -w $toplev/testlong-benchmark.log
fi

if test -e $toplev/optlevel-stats.log; then
    rm -f $toplev/optlevel-stats.log
fi

echo "[$optlevel] make all:      `tail -2 $toplev/make.log | fgrep user`"               >> $toplev/optlevel-stats.log
echo "[$optlevel] make test:     `tail -2 $toplev/test-benchmark.log | fgrep user`"     >> $toplev/optlevel-stats.log
echo "[$optlevel] make testlong: `tail -2 $toplev/testlong-benchmark.log | fgrep user`" >> $toplev/optlevel-stats.log
echo "[$optlevel] make test:     `fgrep SUMMARY $toplev/test.log`"                      >> $toplev/optlevel-stats.log
echo "[$optlevel] make testlong: `fgrep SUMMARY $toplev/testlong.log`"                  >> $toplev/optlevel-stats.log

chmod -w $toplev/optlevel-stats.log

cat $toplev/optlevel-stats.log
