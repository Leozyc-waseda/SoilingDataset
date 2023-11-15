/*
Copyright 2010, Ming-Yu Liu

All Rights Reserved 

Permission to use, copy, modify, and distribute this software and 
its documentation for any non-commercial purpose is hereby granted 
without fee, provided that the above copyright notice appear in 
all copies and that both that copyright notice and this permission 
notice appear in supporting documentation, and that the name of 
the author not be used in advertising or publicity pertaining to 
distribution of the software without specific, written prior 
permission. 

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, 
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN 
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 
*/


#pragma once

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <map>


#include <opencv/cxcore.h>
//#include <opencv/highgui.h>
#include "LFLineSegment.h"


#define LF_NUM_LAYER 2
using namespace std;

class LFLineFitter
{
public:
	LFLineFitter();
	~LFLineFitter();

	void Init();

	void Configure(
		double sigmaFitALine,
		double sigmaFindSupport, 
		double maxGap,
		int nLayer,
		int *nLinesToFitInStage,
		int *nTrialsPerLineInStage);

	void Configure(const char *fileName);

	void FitLine(IplImage *inputImage);

	void DisplayEdgeMap(IplImage *inputImage,const char *outputImageName=NULL);
	void SaveEdgeMap(const char *filename);


	int	rWidth() {return width_;};
	int rHeight() {return height_;};
	int rNLineSegments() {return nLineSegments_;};
	LFLineSegment* rOutputEdgeMap() {return outEdgeMap_;};


private:

	int FitALine(const int nWindPoints,CvPoint *windPoints,const double sigmaFitALine,CvPoint2D64f &lnormal);
	int SampleAPixel(map<int,CvPoint> *edgeMap,IplImage *inputImage,int nPixels);
	void FindSupport(const int nWindPoints,CvPoint *windPoints,CvPoint2D64f &lnormal,double sigmaFindSupport,double maxGap,LFLineSegment &ls,CvPoint *proposedKillingList,int &nProposedKillingList,int x0,int y0);
	void Find(int x0,int y0,CvPoint *windPoints,int &nWindPoints,IplImage *inputImage,int localWindSize);
	void Find(map<int,CvPoint> *edgeMap,int x0,int y0,CvPoint *windPoints,int &nWindPoints,IplImage *inputImage,int localWindSize);
	void ISort(double* ra, int nVec, int* ira);
	void SafeRelease();

private:
	int width_;
	int height_;

	// Output
	LFLineSegment *outEdgeMap_;
	int nLineSegments_;
	int nInputEdges_;


	// Fitting parameters
	int nLinesToFitInStage_[LF_NUM_LAYER];
	int nTrialsPerLineInStage_[LF_NUM_LAYER];
	double sigmaFitALine_;
	double sigmaFindSupport_;
	double maxGap_;
	int minLength_;


	// Program parameters
	int nMaxWindPoints_;
	int nMinEdges_;
	int localWindSize_;
	int smallLocalWindowSize_;

	// temporary storage use
	CvPoint *rpoints_;
	double *rProjection_;
	double *absRProjection_;
	int *idx_;
};
