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
#include <opencv/cxcore.h>


class LMDirectionalIntegralDistanceImage
{
public:
	LMDirectionalIntegralDistanceImage();
	~LMDirectionalIntegralDistanceImage();

	void CreateImage(int width, int height);
	void Construct(IplImage *image, float dx, float dy);
	inline float Sum(int x1, int y1, int x2, int y2, int& count);

private:
	void SafeRelease();
	void ComputeIndices();
	void ComputeII(IplImage* image);


	IplImage *iimage_;
	float ds_;
	int xindexed_;
	int* indices_;
	float factor_;
	int width_;
	int height_;

	friend class LMDistanceImage;
};


inline float LMDirectionalIntegralDistanceImage::Sum(int x1, int y1, int x2, int y2, int& count)
{
	double value = -1;

	if (xindexed_)
	{
		if (x1 <= x2)
		{
      int qy = y1+indices_[x2]-indices_[x1];
      int qx = x2;
      int qy1 = y1-indices_[x1]+indices_[x1-1];
      int qx1 = x1-1;

      if (qy >= 0 &&
          qy < height_ &&
          qx >= 0 &&
          qx < width_ &&
          qy1 >= 0 &&
          qy1 < height_ &&
          qx1 >= 0 &&
          qx1 < width_ )
      {
        value = 
          + cvGetReal2D(iimage_,qy, qx)
          - cvGetReal2D(iimage_,qy1, qx1);
      } else {
        value = -1; //width_*height_;
      }
				
			count = x2-x1+1;
		}
		else
		{
      int qy = y2+indices_[x1]-indices_[x2];
      int qx = x1;
      int qy1 = y2-indices_[x2]+indices_[x2-1];
      int qx1 = x2-1;
      if (qy >= 0 &&
          qy < height_ &&
          qx >= 0 &&
          qx < width_ &&
          qy1 >= 0 &&
          qy1 < height_ &&
          qx1 >= 0 &&
          qx1 < width_ )
      {
			value = 
				+ cvGetReal2D(iimage_, qy, qx)
				- cvGetReal2D(iimage_, qy1, qx1);			
			count = x1-x2+1;
      } else {
        value = -1; //width_*height_;
      }
		}

	}
	else
	{
		if (y1 <= y2)
		{
      int qy = y2;
      int qx =x1+indices_[y2]-indices_[y1];
      int qy1 = y1-1;
      int qx1 = x1-indices_[y1]+indices_[y1-1];
      if (qy >= 0 &&
          qy < height_ &&
          qx >= 0 &&
          qx < width_ &&
          qy1 >= 0 &&
          qy1 < height_ &&
          qx1 >= 0 &&
          qx1 < width_ )
      {
			value = 
				+ cvGetReal2D(iimage_, qy, qx)
				- cvGetReal2D(iimage_, qy1, qx1);
      } else {
        value = -1; //width_*height_;
      }
			
			count = y2-y1+1;

		}
		else
		{
      int qy = y1;
      int qx =x2+indices_[y1]-indices_[y2];
      int qy1 = y2-1;
      int qx1 = x2-indices_[y2]+indices_[y2-1];
      if (qy >= 0 &&
          qy < height_ &&
          qx >= 0 &&
          qx < width_ &&
          qy1 >= 0 &&
          qy1 < height_ &&
          qx1 >= 0 &&
          qx1 < width_ )
      {
			value = 
				+ cvGetReal2D(iimage_, qy, qx)
				- cvGetReal2D(iimage_, qy1, qx1);
      } else {
        value = -1; //width_*height_;
      }
			
			count = y1-y2+1;

		}		
	}

	return (float)(value*factor_);
}
