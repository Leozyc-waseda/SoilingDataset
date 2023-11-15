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
#ifndef ICEIMAGEUTILS_H
#define ICEIMAGEUTILS_H

#include "Image/Image.H"
#include "Image/Pixels.H"

#include <Ice/Ice.h>
#include "Ice/ImageIce.ice.H"
#include <jpeglib.h>


using namespace std;
using namespace ImageIceMod;


//Convert an iNVT Image to a datatype acceptable by Ice
template <class T>
ImageIce Image2Ice(const Image<T>& input) {
  ASSERT(input.initialized());

  int size = input.getSize()*sizeof(T);

  ImageIce output;

  output.width  = input.getWidth();
  output.height = input.getHeight();
  output.pixSize = sizeof(T);

  unsigned char* arrayPtr =  const_cast<unsigned char*>
                            (reinterpret_cast<const unsigned char*> (input.getArrayPtr()));

  output.data.resize(size);
  std::copy (arrayPtr, arrayPtr+size, output.data.begin());

  return output;
}

//Convert an Ice image to an iNVT Image
//-- Warning: there is some nasy and potentially dangerous manual marshalling going on here --
//--          We should look for a better solution in the future. --
template <class T>
Image<T> Ice2Image(const ImageIce &input) {
  Image<T> output;
  output.attach((const_cast<T*> (reinterpret_cast<const T* > (&input.data[0]))), input.width, input.height);
  return output.deepcopy();
}
#endif

