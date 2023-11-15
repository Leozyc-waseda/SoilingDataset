/*!@file AppMedia/app-gamma-correction.C do the gamma correction in the HSV space*/

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-gamma-correction.C $
// $Id: app-gamma-correction.C 12962 2010-03-06 02:13:53Z irock $
//

/*! This app is mainly useful to do the gamma correction to the debayered image.
  the reason is that the captured image from XC HD camera is not looks good enough
*/


#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Raster/Raster.H"
#include "Util/log.H"
#include "Util/StringConversions.H"
#include <string>
#include <cstdio>

using namespace std;

int main(int argc, char** argv)
{
  if (argc != 4)
    {
      fprintf(stderr, "USAGE: app-gamma-correction <input color image> "
              "<output corrected color image> <gamma value>");
      return -1;
    }

  string filename = argv[1];
  string outname = argv[2];
  float gamma;
  convertFromString(string(argv[3]), gamma);


  Image<PixRGB<float> > input = Raster::ReadRGB(filename);

  Image<PixHSV<float> > hsvRes = static_cast< Image<PixHSV<float> > > (input/255);

  Image<PixHSV<float> >::iterator itr = hsvRes.beginw();
  while(itr != hsvRes.endw())
    {
      itr->p[2] = pow(itr->p[2], gamma);
      ++itr;
    }

  Image<PixRGB<float> > foutput = static_cast< Image<PixRGB<float> > > (hsvRes);
  foutput = foutput * 255.0F;
  Image<PixRGB<byte> > output = foutput;
  Raster::WriteRGB(output, outname);

  return 0;
}
