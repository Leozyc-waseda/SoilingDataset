#ifndef HOUGHTRANSFORM2_H
#define HOUGHTRANSFORM2_H

#include "MBARI/Geometry2D.H"
#include "Util/Types.H" // for byte

#include <vector>

template <class T> class Image;
template <class T> class PixRGB;

/*******************************************************************************
* PROCEDURE: houghTransform
* PURPOSE: To perform line detection through a hough transform
* NAME: Harold Lim
* DATE: 2006
// thetares is the resolution of each theta
// dRes is the resolution of the D
// returns the number of lines found
*******************************************************************************/
std::vector <LineSegment2D> houghTransform(Image<byte> &inputImage, float thetaRes, float dRes, int threshold, Image< PixRGB<byte> > &output);


#endif
