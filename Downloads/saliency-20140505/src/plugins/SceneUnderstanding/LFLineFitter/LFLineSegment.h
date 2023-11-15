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

Lior Elazary: Modified to work with the INVT
*/



#pragma once
#include <stdio.h>
#define	_USE_MATH_DEFINES
#include <math.h>
#undef	_USE_MATH_DEFINES

#include <opencv/cxcore.h>


class LFLineSegment
{
public:
	LFLineSegment()
  {
    sx_ = 0;
    sy_ = 0;
    ex_ = 0;
    ey_ = 0;
    normal_.x =0;
    normal_.y =0;
    nSupport_ = 0;
    weight_ = 0;
  
  };
	~LFLineSegment() {

  };

	double sx_,sy_,ex_,ey_;
	int nSupport_;
	double len_;
  double weight_;
	CvPoint2D64f normal_;

	inline LFLineSegment& operator=(const LFLineSegment &rhs);

	// For qsort function
	inline static int Compare(const void *l1,const void *l2);
	inline static int LineSegmentCompare( LFLineSegment &l1, LFLineSegment &l2);


	void Read(FILE* fin);	
	void Center(double *center);
	void Translate(double *vec);
	
	double Theta();

	void Rotate(double theta);
	void Scale(double s);
	double Length();

};

inline int LFLineSegment::LineSegmentCompare( LFLineSegment &l1, LFLineSegment &l2 )
{
	if( l1.len_ > l2.len_ )
		return -1;
	else if( l1.len_ == l2.len_ )
		return 0;
	else
		return 1;
};

inline int LFLineSegment::Compare(const void *l1,const void *l2 )
{
	return LineSegmentCompare( *(LFLineSegment*)l1, *(LFLineSegment*)l2 );

};

inline LFLineSegment& LFLineSegment::operator=(const LFLineSegment &rhs)
{
	sx_ = rhs.sx_;
	sy_ = rhs.sy_;
	ex_ = rhs.ex_;
	ey_ = rhs.ey_;
	len_ = rhs.len_;
	weight_ = rhs.weight_;

	nSupport_ = rhs.nSupport_;
	normal_ = rhs.normal_;
	return (*this);
}
