/*!@file BeoSubTaskDecoder.H */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeoSubTaskDecoder.H $
// $Id: BeoSubTaskDecoder.H 8489 2007-06-13 00:59:59Z beobot $
//

#ifndef BEOSUBTASKDECODER_H_DEFINED
#define BEOSUBTASKDECODER_H_DEFINED

#include "Component/ModelManager.H"
#include "GUI/XWindow.H"
#include "Raster/Raster.H"
#include "rutz/shared_ptr.h"
#include "Image/PixelsTypes.H"
#include "Util/readConfig.H"

#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Transforms.H"

#include <vector>
#include <cstdio>
#include <cstdlib>

template <class FLOAT,
          class INT,
          unsigned int SIT_channels> class segmentImageTrackMC;

//! Definition and access functions for the BeoSubCanny
/*! BeoSubCanny is the vision class for submarine. It implements
  the Canny algorithm and can track shape and color*/
class BeoSubTaskDecoder : public ModelComponent
{
public:
  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  BeoSubTaskDecoder(OptionManager& mgr,
                    const std::string& descrName = "BeoSubTaskDecoder",
                    const std::string& tagName = "BeoSubTaskDecoder");

  //! Destructor
  ~BeoSubTaskDecoder();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{
  void setupDecoder(const char* inputColor, bool debug);
  bool runDecoder(ImageSet< PixRGB<byte> > images, float framerate);
  float calculateHz();
  void order();
  //@}

private:

  void setupRed();
  void setupGreen();

  std::vector<bool> frameCounter;
  float fps;
  const char* candidate_color;
  float Hz;
  float res;

  std::vector<float> color;
  std::vector<float> norm;
  std::vector<float> std;
  std::vector<float> adapt;
  std::vector<float> upperBound;
  std::vector<float> lowerBound;

  readConfig colorConf;

  rutz::shared_ptr<XWindow> wini, wino;

  int width, height;

  Image< PixRGB<byte> > ima;
  Image< PixRGB<float> > fima;
  Image< PixRGB<byte> > display;
  Image< PixH2SV2<float> > H2SVimage;

  segmentImageTrackMC<float,unsigned int,4> *segmenter;

  ImageSet< PixRGB<byte> > imgList;
  bool debugmode;
  bool setup;
};

#endif
