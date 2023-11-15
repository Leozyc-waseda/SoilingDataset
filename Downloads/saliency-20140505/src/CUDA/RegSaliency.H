/*!@file Image/RegSaliency.H A class for quick-and-dirty CUDA saliency mapping */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/RegSaliency.H $
// $Id: RegSaliency.H 12962 2010-03-06 02:13:53Z irock $
//

#ifndef REGSALIENCY_H_DEFINED
#define REGSALIENCY_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/PyrBuilder.H"
#include "Image/Pixels.H"
#include "Image/PyramidTypes.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Image/fancynorm.H"

#include "Util/Types.H"

#include <cstdlib>
#include <unistd.h>
#include <list>


// ######################################################################
class RegSaliency : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  RegSaliency(OptionManager& mgr,
             const std::string& descrName = "Regsaliency",
             const std::string& tagName = "Regsaliency");

  //! Destructor
  ~RegSaliency();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  //! Setup up with image to be processed and build the job queue
  void doInput(const Image< PixRGB<byte> > img);

  //! Is the last given input done processing?
  bool outputReady();

  //! Get last computed output
  Image<float> getOutput();

  Image<float> getIMap();

  Image<float> getCMap();

  Image<float> getOMap();

  Image<float> getFMap();

  Image<float> getMMap();

  Image<float> getInertiaMap();

  Image<float> getInhibitionMap();


  //! For internal thread use: Compute a conspicuity map from an image
  void runSaliency();

  //! Calculate the pyramid and post the generated conspicuity map to the saliency output map
  Image<float> postChannel(Image<float> curImage, PyramidType ptyp, float orientation, float weight, Image<float>& outmap);

  // Just process the pyramid to get the conspicuity map using center surround max norm process
  Image<float> processPyramid(ImageSet<float> pyr);

  //! Calculate intensity channel
  void calcIntensity(const Image<PixRGB<float> > & colImage, Image<float>& outmap);

  //! Calculate the color oppponent channel
  void calcColor(const Image<PixRGB<float> > & colImage, Image<float>& outmap);

  //! Calculate a particular orientation channel
  void calcOrientation(const Image<PixRGB<float> > & colImage, float orientation, Image<float>& outmap);

  //! Calculate the flicker channel
  void calcFlicker(const Image<PixRGB<float> >& colImage, Image<float>& outmap);

  //! Calculate the motion channel
  void calcMotion(const Image<PixRGB<float> > & colImage, int motionIndex);
  //@}

protected:

  void start1(); //!< get started
  void stop2();  //!< get stopped

private:
  Image<float> prevLum;
  Image<float> curLum;
  ImageSet<float> curLumPyr;
  ReichardtPyrBuilder<float> **reichardtPyr;
  Image<float> rg;
  Image<float> by;
  Image<float> randBuf;
  Image<PixRGB<float> > colima;
  bool gotLum;
  bool gotRGBY;
  bool gotSaliency;
  Image<float> outmap;
  Image<float> itsInertiaMap;
  Image<float> itsInhibitionMap;
  Image<float> intensityMap,colorMap,orientationMap,flickerMap,motionMap;
  Image<float> convmap;
  int numMotionDirs,numOrientationDirs;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
