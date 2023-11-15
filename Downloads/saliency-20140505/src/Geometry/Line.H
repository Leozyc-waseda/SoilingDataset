/*!@file FeatureMatching/Line.H Line segments algs */

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

#ifndef LINE_H_DEFINED
#define LINE_H_DEFINED

#include "Util/Types.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ImageSet.H"
#include "Image/Point2D.H"
#include "Image/Point3D.H"

#include <vector>
#include <map>

class Line
{

public:
    Line(float sx, float sy, float ex, float ey);
    Line(Point2D<float> inP1, Point2D<float> inP2);
    Line(Point2D<int> inP1, Point2D<int> inP2);

    ~Line() {} ;

    double getOri();
    double getLength();
    void scale(double s);
    void scale(double sx, double sy);
    Point2D<float> getCenter();
    void rotate(double theta);
    void shear(double k1, double k2);
    void trans(Point2D<float> p);

    void quantize(int numDirections);
    void setQuantizedDir(int numDirections);

    void setWeight(double w) { itsWeight = w; }
    double getWeight() {return itsWeight; }
 
    Point2D<float> getP1() { return p1; }
    Point2D<float> getP2() { return p2; }

    int getDirectionIdx() { return itsDirectionIdx; }
    

private:

  Point2D<float> p1;
  Point2D<float> p2;
  int itsDirectionIdx;

  double sum;
  double cost;
  double prob;
  double itsWeight;
  double itsLength;

};


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif //
