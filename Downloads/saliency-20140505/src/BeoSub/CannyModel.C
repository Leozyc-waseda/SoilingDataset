/*!@file BeoSub/CannyModel.C Simple shape models */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/CannyModel.C $
// $Id: CannyModel.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "CannyModel.H"
#include "Image/DrawOps.H"
#include "Image/ColorOps.H"
#include "GUI/XWindow.H"
#include <cmath>

// ######################################################################
ShapeModel::ShapeModel(const int ndims, const double thresh,
                       double* dims, const bool debug) :
  itsNumDims(ndims), itsThreshold(thresh),
  itsDebugMode(debug), itsWindow()
{

  itsDimensions = (double*)calloc(ndims+1, sizeof(double));
  for(int i = 0; i <=ndims; i++){

    itsDimensions[i] = dims[i];
  }

}

// ######################################################################
ShapeModel::~ShapeModel()
{
  itsWindow.reset(NULL);
}

// ######################################################################
double ShapeModel::getThreshold() const
{ return itsThreshold; }

// ######################################################################
int ShapeModel::getNumDims() const
{ return itsNumDims; }

double* ShapeModel::getDimensions() const
{
  double* tempDims = (double*)calloc(itsNumDims+1, sizeof(double));
  for(int i = 0; i < itsNumDims; i++){
    tempDims[i] = itsDimensions[i];
  }

  return itsDimensions;
}

void ShapeModel::setDimensions(double* in)
{
  for(int i = 0; i <=itsNumDims; i++){
    itsDimensions[i] = in[i];
  }
}

// ######################################################################
float ShapeModel::getDistVal(const double x, const double y,
                             const Image<float>& distMap,
                             Image< PixRGB<byte> >& xdisp) const
{
  int xx = int(x + 0.5), yy = int(y + 0.5);

  // if we are inside the image, just get the distance from the
  // distance map:
  if (distMap.coordsOk(int(x), int(y)))
    {
      if (xdisp.initialized())
        drawDisk(xdisp, Point2D<int>(xx, yy), 3, PixRGB<byte>(255, 0, 0));
      float d = distMap.getValInterp(x, y);
      return d * d;
    }
  else
    {
      // we are outside the distance map; return an error that depends
      // on how far outside we are:
      int xerror = 0, yerror = 0;
      if (xx < 0) xerror = -xx;
      else if (xx >= distMap.getWidth()) xerror = xx - distMap.getWidth();

      if (yy < 0) yerror = -yy;
      else if (yy > distMap.getHeight()) yerror = yy - distMap.getHeight();

      return 10.0f * ((xerror + yerror) * (xerror + yerror));//May need a higher penalty than 10. FIX?
    }
}

// ######################################################################
double ShapeModel::calcDist(double p[], const Image<float>& distMap) const
{

  // Prepare an image to draw into:
  Image< PixRGB<byte> > xdisp;
  xdisp = toRGB(Image<byte>(distMap));

  // if this is the first time we are called and we are in debug mode,
  // open an XWindow of the size of the distMap:
  if (itsWindow.get() == NULL && itsDebugMode == true){
    const_cast<ShapeModel *>(this)->
      itsWindow.reset(new XWindow(distMap.getDims()));
    const_cast<ShapeModel *>(this)->
      itsWindow->setPosition((xdisp.getWidth()*2)+20, 0);
  }

  // get the distance:
  double dist = getDist(p, distMap, xdisp);

  // if we have a window, show shape and distance map:
  if (itsWindow.get()){
    itsWindow->drawImage(xdisp);
  }
  return dist;
}

// ######################################################################
RectangleShape::RectangleShape(const double thresh, double* dims, const bool debug) :
  ShapeModel(5, thresh, dims, debug)
{  }

// ######################################################################
RectangleShape::~RectangleShape()
{ }

// ######################################################################
float RectangleShape::getDist(double p[], const Image<float>& distMap,
                              Image< PixRGB<byte> >& xdisp) const
{
  //NEED to put in limitations keeping shape from becoming too small and keeping any part of the shape from leaving the scope of the image! FIX!!

  // ########## Parse the parameters into human form:
  double x_center = p[1];
  double y_center = p[2];
  double alpha = p[3];

  double width = p[4];
  double height = p[5];

  // ########## Trace the shape and accumulate distances:
  float sina = sin(alpha/10.0);
  float cosa = cos(alpha/10.0);
  int numPts = 0;
  float dist = 0.0;

  // The following is code for a free-form rectangle.
  for (int i = -5; i < 5; i++) {
    // (-w/2,-h/2) -> (-w/2, h/2)
    float tempValX = x_center - width*cosa/2.0 - (i+1)*height*sina/10.0;
    float tempValY = y_center - width*sina/2.0 + (i+1)*height*cosa/10.0;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (w/2, -h/2) -> (w/2, h/2)
    tempValX = x_center + width*cosa/2.0 - (i+1)*height*sina/10.0;
    tempValY = y_center + width*sina/2.0 + (i+1)*height*cosa/10.0;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (-w/2, -h/2) -> (w/2, -h/2)
    tempValX = x_center + height*sina/2.0 + (i+1)*width*cosa/10.0;
    tempValY = y_center - height*cosa/2.0 + (i+1)*width*sina/10.0;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (-w/2, h/2) -> (w/2, h/2)
    tempValX = x_center - height*sina/2.0 + (i+1)*width*cosa/10.0;
    tempValY = y_center + height*cosa/2.0 + (i+1)*width*sina/10.0;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;
  }

  dist = dist / numPts;

  // ########## Add a distance penalty if shape too small:
  if (width < 40) dist += (10.0*((40 - width) * (40 - width)));
  if (height < 30) dist += (10.0*((30 - height) * (30-height)));

  return dist;
}


// ######################################################################
SquareShape::SquareShape(const double thresh, double* dims, const bool debug) :
  ShapeModel(4, thresh, dims, debug)
{  }

// ######################################################################
SquareShape::~SquareShape()
{ }

// ######################################################################
float SquareShape::getDist(double p[], const Image<float>& distMap,
                              Image< PixRGB<byte> >& xdisp) const
{
  // ########## Parse the parameters into human form:
  double x_center = p[1];
  double y_center = p[2];
  double alpha = p[4];
  double height = p[3];

  // ########## Trace the shape and accumulate distances:
  float sina = sin(alpha/10.0);
  float cosa = cos(alpha/10.0);
  int numPts = 0;
  float dist = 0.0;

  //Following is the mathematical representation of a square
  for(int i=-5; i<5; i++) {
    // (-h/2,-h/2) -> (-h/2, h/2)
    float tempValX = x_center - height*cosa/2 - (i+1)*height*sina/10;
    float tempValY = y_center - height*sina/2 + (i+1)*height*cosa/10;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (h/2, -h/2) -> (h/2, h/2)
    tempValX = x_center + height*cosa/2 - (i+1)*height*sina/10;
    tempValY = y_center + height*sina/2 + (i+1)*height*cosa/10;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (-h/2, -h/2) -> (h/2, -h/2)
    tempValX = x_center + height*sina/2 + (i+1)*height*cosa/10;
    tempValY = y_center - height*cosa/2 + (i+1)*height*sina/10;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (-h/2, h/2) -> (h/2, h/2)
    tempValX = x_center - height*sina/2 + (i+1)*height*cosa/10;
    tempValY = y_center + height*cosa/2 + (i+1)*height*sina/10;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;
  }

  dist = dist / numPts;

  // ########## Add a distance penalty if shape too small:
  if (height < 30) dist += (10.0 * ((30 - height)*(30 - height)));

  return dist;
}



// ######################################################################
OctagonShape::OctagonShape(const double thresh, double* dims, const bool debug) :
  ShapeModel(4, thresh, dims, debug)
{  }

// ######################################################################
OctagonShape::~OctagonShape()
{ }

// ######################################################################
float OctagonShape::getDist(double p[], const Image<float>& distMap,
                              Image< PixRGB<byte> >& xdisp) const
{
  // ########## Parse the parameters into human form:
  double x_center = p[1];
  double y_center = p[2];
  double alpha = p[4];
  double height = p[3]; //note that "height" may not be quite the correct label here

  // ########## Trace the shape and accumulate distances:
  float sina = sin(alpha/10.0);
  float cosa = cos(alpha/10.0);
  //funky angles for diagonal edges in octagon
  float sinb = sin((alpha+40.00)/10.0);
  float cosb = cos((alpha+40.00)/10.0);
  int numPts = 0;
  float dist = 0.0;


  //Following is the mathematical representation of an octagon
  for(int i=-5; i<5; i++) {
      //Diagonal edges are even-numbered
      // edge 1
      float tempValX = x_center - height*cosa/2 - ((i+1)*height*sina/10)/2.2;
      float tempValY = y_center - height*sina/2 + ((i+1)*height*cosa/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      //edge 2
      tempValX = x_center - height*cosb/2 - ((i+1)*height*sinb/10)/2.2;
      tempValY = y_center - height*sinb/2 + ((i+1)*height*cosb/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      // edge 3
      tempValX = x_center + height*cosa/2 - ((i+1)*height*sina/10)/2.2;
      tempValY = y_center + height*sina/2 + ((i+1)*height*cosa/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      // edge 4
      tempValX = x_center + height*cosb/2 - ((i+1)*height*sinb/10)/2.2;
      tempValY = y_center + height*sinb/2 + ((i+1)*height*cosb/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      // edge 5
      tempValX = x_center + height*sina/2 + ((i+1)*height*cosa/10)/2.2;
      tempValY = y_center - height*cosa/2 + ((i+1)*height*sina/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      // edge 6
      tempValX = x_center + height*sinb/2 + ((i+1)*height*cosb/10)/2.2;
      tempValY = y_center - height*cosb/2 + ((i+1)*height*sinb/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      // edge 7
      tempValX = x_center - height*sina/2 + ((i+1)*height*cosa/10)/2.2;
      tempValY = y_center + height*cosa/2 + ((i+1)*height*sina/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;

      // edge 8
      tempValX = x_center - height*sinb/2 + ((i+1)*height*cosb/10)/2.2;
      tempValY = y_center + height*cosb/2 + ((i+1)*height*sinb/10)/2.2;
      dist += getDistVal(tempValX, tempValY, distMap, xdisp);
      numPts++;
    }

  dist = dist / numPts;

  // ########## Add a distance penalty if shape too small:
  if (height < 20) dist += (10.0 * ((20 - height) * (20 - height)));

  return dist;
}



// ######################################################################
CircleShape::CircleShape(const double thresh, double* dims, const bool debug) :
  ShapeModel(3, thresh, dims, debug)
{  }

// ######################################################################
CircleShape::~CircleShape()
{ }

// ######################################################################
float CircleShape::getDist(double p[], const Image<float>& distMap,
                              Image< PixRGB<byte> >& xdisp) const
{
  // ########## Parse the parameters into human form:
  double x_center = p[1];
  double y_center = p[2];
  double radius = p[3];

  // ########## Trace the shape and accumulate distances:
  int numPts = 0;
  float dist = 0.0;


  //Following is the mathematical representation of a circle
  for(int i=-7; i<7; i++) {
    // The following is code for a circle
    float tempValX = x_center + radius * cos(i*2*M_PI/14);
    float tempValY = y_center + radius * sin(i*2*M_PI/14);
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;
  }

  dist = dist / numPts;

  // ########## Add a distance penalty if shape too small:
  if (radius < 20) dist += (10.0 * ((20 - radius) * (20 - radius)));

  return dist;
}



// ######################################################################
ParallelShape::ParallelShape(const double thresh, double* dims, const bool debug) :
  ShapeModel(5, thresh, dims, debug)
{  }

// ######################################################################
ParallelShape::~ParallelShape()
{ }

// ######################################################################
float ParallelShape::getDist(double p[], const Image<float>& distMap,
                              Image< PixRGB<byte> >& xdisp) const
{

  // ########## Parse the parameters into human form:
  double x_center = p[1];
  double y_center = p[2];
  double alpha = p[3];

  double width = p[4];
  double height = p[5];

  // ########## Trace the shape and accumulate distances:
  float sina = sin(alpha/10.0);
  float cosa = cos(alpha/10.0);
  int numPts = 0;
  float dist = 0.0;

  // The following is code for a free-form rectangle.
  for (int i = -5; i < 5; i++) {

    // (-w/2, -h/2) -> (w/2, -h/2)
    float tempValX = x_center + height*sina/2.0 + (i+1)*width*cosa/10.0;
    float tempValY = y_center - height*cosa/2.0 + (i+1)*width*sina/10.0;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;

    // (-w/2, h/2) -> (w/2, h/2)
    tempValX = x_center - height*sina/2.0 + (i+1)*width*cosa/10.0;
    tempValY = y_center + height*cosa/2.0 + (i+1)*width*sina/10.0;
    dist += getDistVal(tempValX, tempValY, distMap, xdisp);
    numPts++;
  }

  dist = dist / numPts;

  // ########## Add a distance penalty if shape too small:
  if (width < 70) dist += (10.0 * ((70 - width) * (70-width)));
  if (height < 30) dist += (10.0 * ((30 - height) * (30 - height)));

  return dist;
}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
