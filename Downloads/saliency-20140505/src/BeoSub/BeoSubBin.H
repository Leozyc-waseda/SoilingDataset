/*!@file BeoSub/BeoSubBin.C find pipe     */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubBin.H $
// $Id: BeoSubBin.H 9412 2008-03-10 23:10:15Z farhan $


#ifndef BEOSUB_BEOSUBBIN_DEFINED
#define BEOSUB_BEOSUBBIN_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "MBARI/Geometry2D.H"
#include "Image/OpenCVUtil.H"

class BeoSubBin
{
public:

  // ######################################################################
  //! @name Constructor, assigment and destructor
  //@{

  //! Constructor
  BeoSubBin();

  //! Destructor
  ~BeoSubBin();


  struct BinAngles {
    double angle;
    Point2D<int> pos;
  };
  //@}

  // ######################################################################
  //! @name Compute functions
  //@{

  std::vector<LineSegment2D> getHoughLines  (Image< PixRGB<byte> > &image, Image< PixRGB< byte > > &preHough,
                                             Image< PixRGB<byte> > &outputImage);

  void pruneLines(std::vector<LineSegment2D>& lines, std::vector<LineSegment2D>& pruned, Image< PixRGB<byte> > *img);
  void pruneAngles(std::vector<LineSegment2D>& lines, std::vector<BinAngles>& angles, Image< PixRGB< byte > > *img);
  void removeOrangePipe(Image< PixRGB<byte> >& img);
  float getBinSceneMass(Image< PixRGB<byte> > &cameraImage, Image< PixRGB<byte> > &outputImage, Point2D<int> &center);
  void getWeightedBinMass(std::vector<BinAngles>& angles, Point2D<int> & center, bool cropWindow, Image<PixRGB<byte> >* img);
  void getParallelIntersections(std::vector<LineSegment2D>& lines, std::vector<LineSegment2D>& frontLines, Image< PixRGB<byte> >& img);
  void getBinCenter(std::vector<BinAngles>& angles, Point2D<int>& center);

  //}@

  std::vector<BinAngles> binAngles;

private:

  std::vector<BinAngles> imageAngles;
  std::vector<LineSegment2D> imageLines;

  std::vector<LineSegment2D> binLines;

  Point2D<int> binCenter;



};

#endif


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
