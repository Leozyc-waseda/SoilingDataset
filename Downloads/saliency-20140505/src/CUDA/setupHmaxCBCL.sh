#!/bin/bash
# Script to automate prepping and running of neovision2 code
# ./setupHmaxCBCL.sh <input_movie> <labels_file> [<test_movie>]

##
#
if [ $# -lt 2 ] || [ $# -gt 3 ]
then
echo "Usage: $0 <input_movie> <labels_file> [<test_movie>]"
exit 1
fi
DW=`which $0`
DD=`dirname $DW`
BASE=`dirname $DD`
if [ $BASE == '.' ] 
then
BASE='..'
fi

if [ ! -d "${BASE}/saliency" ]; then echo "You need saliency/ in $BASE"; exit 1; fi
if [ ! -d "${BASE}/svm" ]; then echo "You need svm/ (libsvm distro) in $BASE"; exit 1; fi

# CUDA Devices to use for saliency and the display, and for the HMAX server
CUDADISPDEV=0
CUDAHMAXDEV=0

# Movie to extract patches and training data from
INPUTMOV=$1  #/lab/igpu/00136.mts
# Labels of different categories to train for
LABELFILE=$2  #$DD/labels.txt
# Movie to test performance [Optional]
TESTMOV=$3   #/lab/igpu/00137.mts


# Directory where the patches extracted from the input will be stored
SAMPLEDIR=$DD/Samples/
# Input level input patches (they are Gabors)
C0PATCHES=$DD/c0Patches.txt
# Stored C1 Layer activations
C1PATCHES=$DD/c1Patches.txt
# Once the system is complete, it can run a test image
FEATURESFILE=$DD/features.out
# SVM Model & Range files to be learned
SVMMODEL=$DD/train.out.model
SVMRANGE=$DD/train.out.range
# SVM Tools directory where the training program for the classifier is (easy.py)
SVMTOOLSDIR=$BASE/svm/tools
# Patch server port is +1 of label server port
LOCIP=127.0.0.1
LOCALPORT=9930
SRVIP=127.0.0.1
SRVPORT=9931

echo -n "Do you want to save images from your input movie into the samples directory? [y/n]"
read saveimages
if [ "$saveimages" == "y"  ]; then
mkdir -p $SAMPLEDIR
# 1.1) Run a server to extract the salient chips from each frame in the background
"$BASE/saliency/bin/saveimage-server" $SAMPLEDIR $LOCALPORT $SRVIP $SRVPORT > /dev/null &
SISPID=$!
# 1.2) Run neovision to run the saliency process on the movie and feed the saveimage server
"$BASE/saliency/bin/neovision2cuda" $CUDADISPDEV $LOCIP:$LOCALPORT --in=movie:$INPUTMOV --out=display: > /dev/null 2>&1
# 1.3) Extraction complete, end the saveimage server
kill $SISPID

# 1bis) alternatively, extract whole images from which patches will be based
#$BASE/saliency/bin/stream --in=movie:$INPUTMOV --out=pnm:$SAMPLEDIR/

fi

echo -n "Do you want to extract patches from the samples directory? [y/n]"
read extractpatches
if [ "$extractpatches" == "y" ]; then
# 2) Extract patches from samples
"$BASE/saliency/bin/extractcudacbclpatches" $CUDAHMAXDEV $C0PATCHES $C1PATCHES "$SAMPLEDIR" > /dev/null 2>&1
if [ $? -ne 0 ];  then
echo "Extracting patches failed"
exit 1
fi

fi

echo -n "Do you want to create $DD/train, $DD/test, and subdirs based on the labels in $LABELFILE? [y/n]"
read createdirs
if [ "$createdirs" == "y" ]; then
# 3) Create test and training directories
# NOTE: Maybe we should ask to clear out old training/testing chips?
while read line
do
ID=`echo $line | awk '{print $1}' `;
LABEL=`echo $line | awk '{print $2}' `;
mkdir -p "$DD/Train/$LABEL"
mkdir -p "$DD/Test/$LABEL"
done < $LABELFILE
fi


echo -n "If needed, please organize chips from $SAMPLEDIR into $DD/Train and $DD/Test, press enter when done"
read ignore
# 4) need to manually organize the chips into a bunch of subdirs of $DD/Train and $DD/Test



echo -n "Do you want to calculate features for training and testing data? [y/n]"
read calcfeatures
if [ "$calcfeatures" == "y" ]; then
# 5) Run Cuda HMAX in Batch Mode to Get Example Feature Arrays to train the SVM
rm -f $DD/train.out
rm -f $DD/test.out
while read line
do
ID=`echo $line | awk '{print $1}' `;
LABEL=`echo $line | awk '{print $2}' `;
$BASE/saliency/bin/runcudacbcl $CUDAHMAXDEV $C0PATCHES $C1PATCHES dir:"$DD/Train/$LABEL/" $ID "$DD/$LABEL.out" > /dev/null 2>&1
if [ $? -ne 0 ];  then
echo "Extracting feature arrays for training set failed"
exit 1
fi
cat "$DD/$LABEL.out" >> "$DD/train.out"
rm -f $DD/$LABEL.out
$BASE/saliency/bin/runcudacbcl $CUDAHMAXDEV $C0PATCHES $C1PATCHES dir:"$DD/Test/$LABEL/" $ID "$DD/$LABEL.out" > /dev/null 2>&1
if [ $? -ne 0 ];  then
echo "Extracting feature arrays for testing set failed"
exit 1
fi
cat "$DD/$LABEL.out" >> "$DD/test.out"
rm -f $DD/$LABEL.out
done < $LABELFILE
fi

echo -n "Do you want to train the SVM on the training and testing data? [y/n]"
read trainsvm
if [ "$trainsvm" == "y" ]; then
# 6) Run SVM Training (must be done in SVMTOOLS dir, because it is crappy and locally includes things)
ORIGDIR=`pwd`
cp -f train.out test.out $SVMTOOLSDIR/
cd $SVMTOOLSDIR
python easy.py train.out test.out > /dev/null 2>&1
if [ $? -ne 0 ];  then
echo "SVM training failed"
exit 1
fi

cp -f train.out.model $DD
cp -f train.out.range $DD
cd $ORIGDIR
fi

echo -n "Do you want to test the system performance using the movie $TESTMOV? [y/n]"
read testmovie
if [ "$testmovie" == "y" ]; then
# 7.1) run a server to extract only the windowed sections that are given from saliency as a basis for extracting patches
# Run HMAX Server In Test Mode To Predict Labels
"$BASE/saliency/bin/cudacbcl-server" $CUDAHMAXDEV $LABELFILE $C0PATCHES $C1PATCHES $FEATURESFILE $LOCALPORT $SRVIP $SRVPORT $SVMMODEL $SVMRANGE &
HMSPID=$!
# 7.2) Run neovision
"$BASE/saliency/bin/neovision2cuda" $CUDADISPDEV $LOCIP:$LOCALPORT --in=movie:$TESTMOV --out=display:
# 7.3) Extraction complete, end the save image server
kill $HMSPID

# 7bis) Alternatively, run Cuda HMAX Batch Tester 
#while read line
#do
#ID=`echo $line | awk '{print $1}' `;
#LABEL=`echo $line | awk '{print $2}' `;
#$BASE/saliency/bin/testcudacbcl $CUDAHMAXDEV $C0PATCHES $C1PATCHES dir:$DD/Test/$LABEL/ $SVMMODEL $SVMRANGE $LABEL.test.out
#done < $LABELFILE

fi



