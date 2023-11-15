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
#include <iostream>
#include <iomanip>
#include <string>
#include <stdio.h>
#include "plugins/SceneUnderstanding/EIEdgeImage/EIEdgeImage.h"
#include "plugins/SceneUnderstanding/LFLineFitter/LFLineFitter.h"
#include "LMDistanceImage.h"



class LMLineMatcher {
public:
	LMLineMatcher();
	~LMLineMatcher();

  struct Rect
  {
    int x;
    int y;
    int width;
    int height;
    double distance;

    Rect(int inX, int inY, int w, int h, double d) :
      x(inX), y(inY), width(w), height(h), distance(d)
    {}
  };

	// Initialization
	void Configure(const char* fileName);
	void Init(const char* fileName,double db_scale=1.0);
	//void Match(LFLineFitter &lf);
  std::vector<Rect> Match(int width, int height, int nLines, LFLineSegment* linesSegment);

	void GetColorImage(string filename) {imageName_ = filename;};

  double getCost(EIEdgeImage& tbImage, double ltrans[2], double factor, int& count);
  double getCost(int directionIdx, int sx, int sy, int ex, int ey, int count) const;
  
  void computeIDT3(int width, int height, int nLines, LFLineSegment* linesSegment);
  

private:

	double MatchBruteForce(EIEdgeImage& dbImage, int index, int* iindices, int* indices, int* xIndex, int* yIndex, double* dIndex, double* sIndex, double* distances, int& counter, double& minCost);
	void DrawDetWind(IplImage *image,int x,int y,int detWindWidth,int detWindHeight,CvScalar scalar=cvScalar(0,255,0),int thickness=2);
	void DrawMatchTemplate(IplImage *image,EIEdgeImage &ei,int x,int y,double scale,CvScalar scalar=cvScalar(0,255,255),int thickness=1);
	void ISort(double* ra, int nVec, int* ira);
	void SafeRelease();


private:

	// temporary structures
	double* distances_;
	int*	indices_;
	int*	xIndices_;
	int*	yIndices_;
	double* sIndices_;
	double*	dIndices_;
	int*	iindices_;
	int*	wIndices_;
	int*	hIndices_;
	double scale_;
	EIEdgeImage* dbImages_;

	// tempaltes
	int ndbImages_;
	LMDistanceImage queryDistanceImage_;

	// test image
	string imageName_;
	EIEdgeImage	queryImage_;


	// Matching parameter	
	int nDirections_;	
	double maxCost_;
	float directionCost_;

	double db_scale_;

	double baseSearchScale_;
	double minSearchScale_;
	double maxSearchScale_;

	int searchStepSize_;
	int searchBoundarySize_;
	double minCostRatio_;
};
