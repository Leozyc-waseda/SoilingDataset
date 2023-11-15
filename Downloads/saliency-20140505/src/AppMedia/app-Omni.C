/*!@file AppMedia/app-Omni.C  Omnidirectional lens distortion correction tool
 */

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
// Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-Omni.C $
// $Id: app-Omni.C 6410 2006-04-01 22:12:24Z rjpeters $
//

#include "Image/Image.H"
#include "Image/OmniOps.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/readConfig.H"

// ############################################################
// ############################################################
// ##### ---OMNI---
// ##### OMNI DIRECTIONAL CORRECTION TOOL:
// ##### T. Nathan Mundhenk nathan@mundhenk.com
// ############################################################
// ############################################################

//! omni directional correction tool
template<class T> class Omni
{
public:
  //!Omni correction object constructor
  Omni();
  //!Omni correction object destructor
  ~Omni();
  //! Run omni correction on an Image, return a new Image
  /*! This method will call both Omni corrections for
    general and special correction on an Image. The two
    methods can be found in Image.H as omniCorrectGen and
    omniCorrectSP. If a pinhole image is used omniCorrectSP
    may be skipped.
    @param image This is the image to be corrected of type Image
    @param config This is the config file for processing of type readConfig
  */
  Image<T> run(const Image<T>& image, readConfig &config);
private:
  int centerDetect,radiusDetect,centerX,centerY,radX,radY,radAdg,nebRadius;
  float r,h,k;
  Image<T> correctImage;
};

//! This is the configFile name
const char* configFile;
//! This is the configFile object
readConfig configIn(25);
//! This is an Image object for RGB images
Image< PixRGB<byte> > image;
//! This is an Image object for RGB images
Image< PixRGB<byte> > newImage;
//! This is an Image object for RGB images
Omni< PixRGB<byte> > omni;
/*! This is the main funcion for omni directional correction.
  Its main purpose is to call the omni-directional methods
  in class Omni. Call from src2. You must supply Omni.conf
  as a config file. The command line argument is
  "../bin/Omni ../image_in/imagefile". Images centers and
  radius can be calculated by Omni, or specified in the
  Omni.conf file.
*/


int main(int argc, char* argv[])
{
  configFile = "Omni.conf";
  configIn.openFile(configFile);
  if(argc > 1)
  {
    newImage.resize(1,1);
    image = Raster::ReadRGB(argv[1], RASFMT_PNM);
    Raster::VisuRGB(image,"infile.ppm");
    newImage = omni.run(image,configIn);
    Raster::VisuRGB(newImage,"outfile.ppm");
  }
  return 0;
}

template <class T>
Omni<T>::Omni()
{
}

template <class T>
Omni<T>::~Omni()
{
}

//! Run omni directional equations on an Image with parameters from readConfig
template <class T>
Image<T> Omni<T>::run(const Image<T>& image, readConfig &config)
{
  nebRadius = (int)config.getItemValueF("centerDetect");
  centerDetect = (int)config.getItemValueF("centerDetect");
  LINFO("centerDetect %d", centerDetect);
  radiusDetect = (int)config.getItemValueF("radiusDetect");
  LINFO("radiusDetect %d", radiusDetect);
  r = config.getItemValueF("r");
  LINFO("r %f",r);
  h = config.getItemValueF("h");
  LINFO("h %f",h);
  k = config.getItemValueF("k");
  LINFO("k %f",k);
  if(centerDetect == 1)
  {
    centerX = (image.getWidth()/2);
    centerY = (image.getWidth()/2);
  }
  else
  {
    centerX = (int)config.getItemValueF("centerX");
    centerY = (int)config.getItemValueF("centerY");
  }

  if(radiusDetect == 1)
  {
    radAdg = (int)config.getItemValueF("radAdg");
    radX = (image.getWidth()/2);
    radY = radX;
  }
  else
  {
    radX = (int)config.getItemValueF("radX");
    radY = (int)config.getItemValueF("radY");
  }
  LINFO("radX %d",radX);
  LINFO("radY %d",radY);
  LINFO("centerX %d",centerX);
  LINFO("centerY %d",centerX);

  LINFO("CORRECTING IMAGE GENERAL");
  Image<T> correctedImage = omniCorrectGen(image,radX,radY,
                                           centerX,centerY,radAdg);
  Raster::VisuRGB(correctedImage,"Omni_general.ppm");
  return omniDenebulize(correctedImage, nebRadius);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
