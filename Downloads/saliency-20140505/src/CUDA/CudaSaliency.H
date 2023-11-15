/*!@file CUDA/CudaSaliency.H A class for quick-and-dirty CUDA saliency mapping */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/CudaSaliency.H $
// $Id: CudaSaliency.H 13326 2010-04-27 22:02:05Z dparks $
//

#ifndef CUDASALIENCY_H_DEFINED
#define CUDASALIENCY_H_DEFINED

#include "Component/ModelComponent.H"
#include "Component/ModelParam.H"
#include "CUDA/CudaColorOps.H"
#include "CUDA/CudaImage.H"
#include "CUDA/CudaImageSet.H"
#include "CUDA/CudaMathOps.H"
#include "Image/DrawOps.H"
#include "CUDA/CudaFilterOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidTypes.H"

#include "Util/Types.H"

#include <cstdlib>
#include <unistd.h>
#include <list>


template <class T> class CudaReichardtPyrBuilder;

// ######################################################################
class CudaSaliency : public ModelComponent
{
public:

  // ######################################################################
  /*! @name Constructors and Destructors */
  //@{

  //! Constructor
  CudaSaliency(OptionManager& mgr,
             const std::string& descrName = "CudaSaliency",
             const std::string& tagName = "CudaSaliency");

  //! Destructor
  ~CudaSaliency();

  //@}

  // ######################################################################
  /*! @name member functions */
  //@{

  virtual void paramChanged(ModelParamBase* const param,
                            const bool valueChanged,
                            ParamClient::ChangeStatus* status);


  //! Setup up with image to be processed and build the job queue
  void doInput(const Image< PixRGB<byte> > img);

  void doCudaInput(const CudaImage< PixRGB<float> > img);

  void setDevice(MemoryPolicy mp_in, int dev_in);

  // Clear stored maps
  void clearMaps();

  //! Is the last given input done processing?
  bool outputReady();

  //! Get last computed output
  Image<float> getOutput();

  CudaImage<float> getCudaOutput();

  CudaImage<float> getIMap();

  CudaImage<float> getCMap();

  CudaImage<float> getOMap();

  CudaImage<float> getFMap();

  CudaImage<float> getMMap();

  CudaImage<float> getInertiaMap();

  CudaImage<float> getInhibitionMap();

  std::vector<Point2D<int> > getSalMaxLoc();

  std::vector<float> getSalMax();

  double getPatchSize();

  //! For internal thread use: Compute a conspicuity map from an image
  void runSaliency();

  //! Calculate the pyramid and post the generated conspicuity map to the saliency output mapt
  CudaImage<float> cudaPostChannel(CudaImage<float> curImage, PyramidType ptyp, float orientation, float weight, CudaImage<float>& outmap);

  // Generate a conspicuity map from a pyramid
  CudaImage<float> processPyramid(CudaImageSet<float> pyr);

  //! Calculate intensity channel
  void calcIntensity(const CudaImage<PixRGB<float> > & colImage, CudaImage<float>& outmap);

  //! Calculate the red/green & blue/yellow color oppponent channel
  void calcColor(const CudaImage<PixRGB<float> > & colImage, CudaImage<float>& outmap);

  //! Calculate a particular orientation channel
  void calcOrientation(const CudaImage<PixRGB<float> > & colImage, float orientation, CudaImage<float>& outmap);

  //! Calculate the flicker channel
  void calcFlicker(const CudaImage<PixRGB<float> >& colImage, CudaImage<float>& outmap);

  //! Calculate the motion channel
  void calcMotion(const CudaImage<PixRGB<float> > & colImage, int motionIndex);

  //! Calculate inhibition of return on the saliency map
  void calcInhibition(const CudaImage<float> & salMap);

  //! Calculate inertia on the saliency map
  void calcInertia(const CudaImage<float> & salMap);

  //@}

protected:

  void start1(); //!< get started
  void stop2();  //!< get stopped

private:
  OModelParam<int> itsCudaDevice;
  Point2D<int> itsInertiaLoc;
  double itsCurrentInertiaFactor;
  CudaImage<float> itsInertiaMap;
  CudaImage<float> itsInhibitionMap;
  OModelParam<int> itsNMostSalientLoc;
  OModelParam<double> itsInertiaHalfLife;
  OModelParam<double> itsInertiaStrength;
  OModelParam<double> itsInertiaRadius;
  OModelParam<double> itsInertiaShiftThresh;
  OModelParam<double> itsIorHalfLife;
  OModelParam<double> itsIorStrength;
  OModelParam<double> itsIorRadius;
  OModelParam<double> itsPatchSize;
  double itsDynamicFactor;
  std::vector<Point2D<int> > itsSalMaxLoc;
  std::vector<float> itsSalMax;
  CudaImage<float> prevLum;
  CudaImage<float> curLum;
  CudaImageSet<float> curLumPyr;
  CudaReichardtPyrBuilder<float> *reichardtPyr;
  CudaImage<float> rg;
  CudaImage<float> by;
  CudaImage<float> randBuf;
  CudaImage<PixRGB<float> > colima;
  bool gotLum;
  bool gotRGBY;
  bool gotSaliency;
  CudaImage<float> outmap;
  CudaImage<float> intensityMap,colorMap,orientationMap,flickerMap,motionMap;
  CudaImage<float> convmap;
  MemoryPolicy mp;
  int dev;
  int numMotionDirs,numOrientationDirs;

};

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
