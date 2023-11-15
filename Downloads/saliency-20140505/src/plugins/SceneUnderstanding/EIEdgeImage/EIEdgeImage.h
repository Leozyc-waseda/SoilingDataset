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
#include "plugins/SceneUnderstanding/LFLineFitter/LFLineSegment.h"
#include "plugins/SceneUnderstanding/LFLineFitter/LFLineFitter.h"
#include <vector>
using namespace std;



class EIEdgeImage {
public:
	EIEdgeImage();
	~EIEdgeImage();
	void SafeRelease();	
	void Read(char* fileName);
	void Read(LFLineFitter &lf);
  void Read(int width, int height, int nLines, LFLineSegment* linesSegment);
  

	void SetNumDirections(const int nDirections) {nDirections_=nDirections;};
	void Scale(double s);
  void setWeights();

	void ConstructDirectionImage(int index,IplImage* image);
	double Length();

	void operator=(EIEdgeImage& ei);
	void Boundary(double &minx, double &miny, double &maxx, double &maxy);
	void SetDirectionIndices();


	// Display
	void ConstructImage(IplImage *image, int thickness = 1);

	int width_;
	int height_;
	int	nLines_;
	LFLineSegment* lines_;
	int* directionIndices_;
  double length;

private:

	void SetLines2Grid();
	void SetDirections();
	int Theta2Index(double theta);
	double Index2Theta(int index);

private:
	int nDirections_;

	vector<LFLineSegment*>* directions_;


	friend class LMLineMatcher;
	friend class LMDistanceImage;
};
