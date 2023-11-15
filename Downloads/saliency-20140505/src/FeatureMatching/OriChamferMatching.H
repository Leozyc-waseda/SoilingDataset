/*!@file FeatureMatching/OriChamferMatching.H Oriented chamfer matching algs */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: $
// $Id: $
//

#ifndef ORICHAMFERMATCHING_H_DEFINED
#define ORICHAMFERMATCHING_H_DEFINED

#include "Image/OpenCVUtil.H"
#include "Util/Types.H"
#include "GUI/DebugWin.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Image/Point2D.H"
#include "Image/Point3D.H"
#include "FeatureMatching/Polygon.H"

#include <vector>
#include <map>


class OriIntegralImage
{

public:
  OriIntegralImage() {}

  OriIntegralImage(const Image<float>& img, float dx, float dy);

  inline float getCost(Point2D<int> p1, Point2D<int> p2) const
  {
    int x1 = p1.i;
    int y1 = p1.j;
    int x2 = p2.i;
    int y2 = p2.j;

    int qy = -1;
    int qx = -1;
    int qy1 = -1;
    int qx1 = -1;

    if (itsXindexed)
    {
      if (x1 <= x2)
      {
        qy = y1 + itsIndices[x2] - itsIndices[x1];
        qx = x2;
        qy1 = y1 - itsIndices[x1] + itsIndices[x1-1];
        qx1 = x1-1;
      } else {
        qy = y2+itsIndices[x1]-itsIndices[x2];
        qx = x1;
        qy1 = y2-itsIndices[x2]+itsIndices[x2-1];
        qx1 = x2-1;
      }
    } else {
      if (y1 <= y2)
      {
        qy = y2;
        qx =x1+itsIndices[y2]-itsIndices[y1];
        qy1 = y1-1;
        qx1 = x1-itsIndices[y1]+itsIndices[y1-1];
      } else {
        qy = y1;
        qx =x2+itsIndices[y1]-itsIndices[y2];
        qy1 = y2-1;
        qx1 = x2-itsIndices[y2]+itsIndices[y2-1];
      }
    }

    double value = -1;
    if (itsIntegralImage.coordsOk(qx,qy) &&
        itsIntegralImage.coordsOk(qx1,qy1))
      value = + itsIntegralImage.getVal(qx,qy) - itsIntegralImage.getVal(qx1, qy1);
    else
      value = -1;

    return (float)(value*itsFactor);
  }

private:
	float itsFactor;
	float itsDS;
	int itsXindexed;
  Image<float> itsIntegralImage;
  std::vector<int> itsIndices;
  
};

class OriChamferMatching 
{

public:
  OriChamferMatching();

  OriChamferMatching(const std::vector<Line>& inLines,
      const int numDirections,
      const double oriCost,
      const Dims dims);

  void setLines(const std::vector<Line>& inLines,
      const int numDirections,
      const double oriCost,
      const Dims dims);


  //! Update the costs using the orientation cost
  void updateOriCost(const double oriCost);

  //! Build integral images from the distance images
  void buildIntegralDistances();

  inline float getCostFast(uint oriIdx, const Point2D<int> p1, const Point2D<int> p2)
  {
    if (oriIdx < itsOriIntDistImages.size())
      return itsOriIntDistImages[oriIdx].getCost(p1,p2);
    else
      return -1;
  }

  inline float getCost(uint oriIdx, const Point2D<int> p1, const Point2D<int> p2)
  {

    if (oriIdx >= itsOriIntDistImages.size())
      return -1;

    if (false)
    {
      LINFO("%ix%i %ix%i ori=%f", p1.i, p1.j, p2.i, p2.j, oriIdx*M_PI/60);
      Image<PixRGB<byte> > tmp = itsOriDistImages[oriIdx];
      drawLine(tmp, p1, p2, PixRGB<byte>(255,0,0));
      SHOWIMG(tmp);
      SHOWIMG(itsOriDistImages[oriIdx]);
    }


    float sum = -1;
    // from Graphics Gems / Paul Heckbert
    int dx = p2.i - p1.i, ax = abs(dx) << 1, sx = signOf(dx);
    int dy = p2.j - p1.j, ay = abs(dy) << 1, sy = signOf(dy);
    int x = p1.i, y = p1.j;

    const int w = itsOriDistImages[oriIdx].getWidth();
    const int h = itsOriDistImages[oriIdx].getHeight();

    float* const sptr = itsOriDistImages[oriIdx].getArrayPtr();

    if (ax > ay)
    {
      int d = ay - (ax >> 1);
      for (;;)
      {
        if (x >= 0 && x < w && y >= 0 && y < h)
        {
          if (sum == -1) 
            sum = sptr[x + w * y];
          else
            sum += sptr[x + w * y];
        }
        if (x == p2.i) return sum;
        if (d >= 0) { y += sy; d -= ax; }
        x += sx; d += ay;
      }
    }
    else
    {
      int d = ax - (ay >> 1);
      for (;;)
      {
        if (x >= 0 && x < w && y >= 0 && y < h)
        {
          if (sum == -1) 
            sum = sptr[x + w * y];
          else
            sum += sptr[x + w * y];
        }
        if (y == p2.j) return sum;
        if (d >= 0) { x += sx; d -= ay; }
        y += sy; d += ax;
      }
    }


    return sum;
  }

 
private:
  std::vector<Line> itsLines;
  ImageSet<float> itsOriDistImages;
  std::vector<OriIntegralImage> itsOriIntDistImages;
  
  bool itsUseOpenCV;
  float itsMaxCost;

  
};



// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
