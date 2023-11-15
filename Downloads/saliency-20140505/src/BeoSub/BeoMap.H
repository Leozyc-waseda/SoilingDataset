/*!@file BeoSub/BeoMap.H Class for stitching images */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2003   //
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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoMap.H $
// $Id: BeoMap.H 6526 2006-04-25 17:26:17Z rjpeters $
//


#ifndef BEOMAP_H_DEFINED
#define BEOMAP_H_DEFINED

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Raster/Raster.H"
#include "SIFT/VisualObject.H"
#include "SIFT/VisualObjectMatch.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

class VisualCortex;
class VisualObjectDB;
class ShapeModel;

typedef struct{
  int leftedge;
  int rightedge;
  int upperedge;
  int loweredge;
}mapedge;

typedef struct{
  int min;
  int max;
}stitchminmax;


//! Definition and access functions for the BeoMap base class
/*! BeoMap is a utility class for stitching/mosaicing images */
class BeoMap
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoMap(float cut_thresh, int size_thresh, bool toCut);

  //! Destructor
   ~BeoMap();

  //@}

  //! method to be called for stitching
  void makePanorama(const char* nam1,const char* nam2);

private:
  float CUT_THRESHOLD;
  int  SIZE_THRESH;
  int globalcounter, pw, ph, switchcnt;
  mapedge medge;
  stitchminmax coordx[10];
  stitchminmax coordy[10];
  bool cut;
  Image< PixRGB<byte> > im3;


};


#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */


