/*!@file BeoSub/ColorTracker.H A class for simple color tracking */

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
// Primary maintainer for this file: Zack Gossman <gossman@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/ColorTracker.H $
// $Id: ColorTracker.H 6990 2006-08-11 18:13:51Z rjpeters $
//

#ifndef COLORTRACKER_H_DEFINED
#define COLORTRACKER_H_DEFINED
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Raster/Raster.H"
#include "VFAT/segmentImageTrackMC.H"
#include "Component/ModelManager.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "rutz/shared_ptr.h"

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


//! Definition and access functions for ColorTracker class
/*! ColorTracker is the color tracking module for the BeoSub. It simply checks for the existence of a color within an image and returns the position of the largest "blob" of that color if found*/
class ColorTracker : public ModelComponent
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  ColorTracker(OptionManager& mgr,
              const std::string& descrName = "ColorTracker",
              const std::string& tagName = "ColorTracker");

  //! Destructor
  ~ColorTracker();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{
  void setupTracker(const char* colorArg, Image< PixRGB<byte> > image, bool debug);
  bool runTracker(float threshold, float &xpos, float &ypos, float &mass);
  //@}

private:

  bool debugmode;    //determines screen output

  rutz::shared_ptr<XWindow> xwin1, xwin2;

  Image< PixRGB<byte> > colorImg;
  Image< PixRGB<byte> > outputImg;

  std::vector<float> color;

  segmentImageTrackMC<float,unsigned int, 4> *segmenter;

  bool hasSetup;
  bool hasRun;
};

#endif
