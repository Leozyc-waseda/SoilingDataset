/*!@file Ice/IceImageUtils.H A set of utilities for using iNVT Images
   with Ice*/
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
// Primary maintainer for this file: Randolph Voorhies <voorhies@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Ice/IceImageUtils.H $
// $Id: IceImageUtils.H$
//
#ifdef HAVE_OPENCV_CV_H
#include "Image/OpenCVUtil.H"
#endif

#ifndef ICEOPENCVUTILS_H
#define ICEOPENCVUTILS_H 

#include <Ice/Ice.h>
#include "Ice/ImageIce.ice.H"
#include <jpeglib.h>
#include "Util/log.H" //For LFATAL

using namespace std;
using namespace ImageIceMod;



#ifdef HAVE_OPENCV_CV_H
//Convert an OpenCV Image to a datatype acceptable by Ice
ImageIce Ipl2Ice(const IplImage* img)
{

  if (img->depth != IPL_DEPTH_8U)
    LFATAL("IplImage must have depth==IPL_DEPTH_8U"
           "for conversion to IceImage/Image<PixRGB<byte>>");

  if (img->nChannels != 3)
    LFATAL("IplImage must have nChannels==3"
           "for conversion to IceImage/Image<PixRGB<byte>>");

  if (img->dataOrder != 0)
    LFATAL("IplImage must have dataOrder==0 (interleaved)"
           "for conversion to IceImage/Image<PixRGB<byte>>");

  int size = (img->nChannels) * (img->width)* (img->height);
  ImageIce output;
  output.width = img->width;
  output.height= img->height;
  output.pixSize = img->nChannels;	
  LINFO("ipl width %d ipl height %d size %d",output.width,output.height,size);
  unsigned char* arrayPtr = (unsigned char*)img->imageData;
  output.data.resize(size);

  std::copy(arrayPtr, arrayPtr+size,output.data.begin());

  return output; 


}
//Convert an Ice image to an OpenCV Image
//-- Warning: there is some nasy and potentially dangerous manual marshalling going on here --
//--          We should look for a better solution in the future. --
template <class T>
IplImage* Ice2Ipl(const ImageIce &input) {
	IplImage* ret_img = cvCreateImageHeader(cvSize(input.width,input.height),IPL_DEPTH_8U, 3);	
	ret_img->imageData = (char*) input.data;
	return ret_img;
}
#endif

#endif
