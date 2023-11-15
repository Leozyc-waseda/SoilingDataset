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


#include "EIEdgeImage.h"

EIEdgeImage::EIEdgeImage()
{
	lines_ = NULL;
	directions_ = NULL;
	directionIndices_	= NULL;
}

EIEdgeImage::~EIEdgeImage()
{
	SafeRelease();
}

void EIEdgeImage::SafeRelease()
{
	if (lines_)
	{
    //delete[] lines_;
		//lines_ = NULL;
	}
	if (directions_)
	{
		for (int i=0 ; i<nDirections_ ; i++)
		{
			directions_[i].clear();
		}
		delete[] directions_;
		directions_ = NULL;
	}

	if (directionIndices_)
		delete[] directionIndices_;
	directionIndices_ = NULL;
};

void EIEdgeImage::Read(char* fileName)
{
  printf("Reading edhe image %s\n", fileName);
	FILE* fin=NULL;
	fin = fopen(fileName, "r");
	if(fileName==NULL)
	{
		cerr<<"[ERROR] Cannot read file "<<fileName<<"\n!!!";
		exit(0);
	}
  int ret=0;
	ret=fscanf(fin, "%d %d", &width_, &height_);
	ret=fscanf(fin, "%d", &nLines_);
	lines_ = new LFLineSegment[nLines_];
	for (int i=0 ; i<nLines_ ; i++)
	{
		lines_[i].Read(fin);
	}

	SetLines2Grid();
	SetDirections();

	fclose(fin);
}

void EIEdgeImage::SetLines2Grid()
{	
	double trans[2];
	double theta;
	double dtheta;
	int index;

	for (int i=0 ; i<nLines_ ; i++)
	{
		theta = lines_[i].Theta();

		index = Theta2Index(theta);
		dtheta = Index2Theta(index) - theta;

		lines_[i].Center(trans);
		for(int j=0;j<2;j++)
			trans[j] *= -1;

		lines_[i].Translate(trans);
		lines_[i].Rotate(dtheta);

		for(int j=0;j<2;j++)
			trans[j] *= -1;

		lines_[i].Translate(trans);
	}
}

int EIEdgeImage::Theta2Index(double theta)
{
	return (int) floor ((theta  *nDirections_) / (M_PI+1e-5));
}

double EIEdgeImage::Index2Theta(int index)
{
	return ((index)*M_PI)/nDirections_ + M_PI/(2*nDirections_);
}


void EIEdgeImage::SetDirections()
{
	int index;
	directions_ = new vector<LFLineSegment*>[nDirections_];
	for (int i=0 ; i<nLines_ ; i++)
	{
		index = Theta2Index(lines_[i].Theta());
		directions_[index].push_back(&(lines_[i]));
	}
}


void EIEdgeImage::Scale(double s)
{
	int i;

  length = 0;
	for (i=0 ; i<nLines_ ; i++)
	{
		lines_[i].Scale(s);
    lines_[i].len_ = lines_[i].Length();
    length += lines_[i].len_;
	}	

	width_ = (int)(width_*s);
	height_ = (int)(height_*s);
}

void EIEdgeImage::setWeights()
{

  for (int i=0 ; i<nLines_ ; i++)
    lines_[i].weight_ = lines_[i].len_ / length;
}

void EIEdgeImage::Read(LFLineFitter &lf)
{

	width_ = lf.rWidth();
	height_ = lf.rHeight();
	nLines_ = lf.rNLineSegments();
	LFLineSegment* lineSegmentMap = lf.rOutputEdgeMap();

	lines_ = new LFLineSegment[nLines_];
	for (int i=0 ; i<nLines_ ; i++)
		lines_[i] = lineSegmentMap[i];

	SetLines2Grid();
	SetDirections();

}

void EIEdgeImage::Read(int width, int height, int nLines, LFLineSegment* linesSegment)
{

	width_ = width;
	height_ = height;
	nLines_ = nLines;
  lines_ = linesSegment;
	//LFLineSegment* lineSegmentMap = linesSegment;

	//lines_ = new LFLineSegment[nLines_];
	//for (int i=0 ; i<nLines_ ; i++)
	//	lines_[i] = lineSegmentMap[i];

	SetLines2Grid();
	SetDirections();

}


void EIEdgeImage::ConstructDirectionImage(int index,IplImage* image)
{
	CvPoint pt1, pt2;
	double vec[2];
	
	cvSet(image, cvScalar(255));
	for (unsigned int i=0 ; i<directions_[index].size() ; i++)
	{
		vec[0] = directions_[index][i]->sx_;
		vec[1] = directions_[index][i]->sy_;
		pt1.x = (int)floor(vec[0]);
		pt1.y = (int)floor(vec[1]);

		vec[0] = directions_[index][i]->ex_;
		vec[1] = directions_[index][i]->ey_;
		pt2.x = (int)floor(vec[0]);
		pt2.y = (int)floor(vec[1]);

		cvLine(image, pt1, pt2, cvScalar(0));
	}
}

double EIEdgeImage::Length()
{
	double length = 0;

	int i;

	for (i=0 ; i<nLines_ ; i++)
	{
		length += lines_[i].Length();
	}	

	return length;
}



void EIEdgeImage::operator=(EIEdgeImage& ei)
{
	SafeRelease();

	width_ = ei.width_;
	height_ = ei.height_;
	nLines_ = ei.nLines_;
	nDirections_ = ei.nDirections_;
	lines_ = new LFLineSegment[nLines_];
	for (int i=0; i<nLines_ ; i++)
	{
		lines_[i] = ei.lines_[i];
	}
}

void EIEdgeImage::Boundary(double &minx, double &miny, double &maxx, double &maxy)
{
	minx = miny = 1e+10;
	maxx = maxy = -1e+10;

	for (int i=0 ; i<nLines_ ; i++)
	{
		if (minx > double(lines_[i].sx_))
			minx = double(lines_[i].sx_);
		if (minx > double(lines_[i].ex_))
			minx = double(lines_[i].ex_);

		if (maxx < double(lines_[i].sx_))
			maxx = double(lines_[i].sx_);
		if (maxx < double(lines_[i].ex_))
			maxx = double(lines_[i].ex_);

		if (miny > double(lines_[i].sy_))
			miny = double(lines_[i].sy_);
		if (miny > double(lines_[i].ey_))
			miny = double(lines_[i].ey_);

		if (maxy < double(lines_[i].sy_))
			maxy = double(lines_[i].sy_);
		if (maxy < double(lines_[i].ey_))
			maxy = double(lines_[i].ey_);
	}	
}

void EIEdgeImage::SetDirectionIndices()
{
	if (directionIndices_)
		delete[] directionIndices_;
	directionIndices_ = new int[nLines_];
	for (int i=0 ; i<nLines_ ; i++)
	{
		directionIndices_[i] = Theta2Index(lines_[i].Theta());
	}

}



void EIEdgeImage::ConstructImage(IplImage *image, int thickness)
{
	int i;
	CvPoint pt1, pt2;
	double len;
	cvSet(image, cvScalar(255,255,255));

	for (i=0 ; i<nLines_ ; i++)
	{
		len = lines_[i].Length();

		if(len>0 )
		{
			pt1.x = (int)ceil(lines_[i].sx_ - 0.5);
			pt1.y = (int)ceil(lines_[i].sy_ - 0.5);
			pt2.x = (int)ceil(lines_[i].ex_ - 0.5);
			pt2.y = (int)ceil(lines_[i].ey_ - 0.5);
		}
		cvLine(image, pt1, pt2, cvScalar(0,0,0), thickness);
	}

}
