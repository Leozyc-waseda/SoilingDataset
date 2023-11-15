/*!@file AppMedia/app-pSNR.C Compute peak signalto-noise ratio between two images */

//////////////////////////////////////////////////////////////////////////
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
//////////////////////////////////////////////////////////////////////////
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
//////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-pSNR.C $
// $Id: app-pSNR.C 6191 2006-02-01 23:56:12Z rjpeters $
//

#include "Image/Image.H"
#include "Image/ColorOps.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"

#include <iostream>

/* ###################################################################### */
int main(int argc, char **argv)
{
  MYLOGVERB = LOG_INFO;

  if (argc < 3)
    {
      std::cerr<<"USAGE: "<<argv[0]<<
        " <image1.ppm> <image2.ppm> [weight.pgm]"<<std::endl;
      exit(1);
    }

  // read input images:
  double psnr;
  Image< PixRGB<byte> > input1 = Raster::ReadRGB(argv[1]);
  Image< PixRGB<byte> > input2 = Raster::ReadRGB(argv[2]);

  std::cout<<argv[1]<<" - "<<argv[2]<<" : ";

  // compute pSNR:
  if (argc > 3)
    {
      Image<byte> w = Raster::ReadGray(argv[3]);
      w.setVal(0, 0, 0);  // make sure w has some zero in it
      Image<float> weight = w; inplaceNormalize(weight, 0.0f, 1.0f);
      psnr = pSNRcolor(input1, input2, weight);
      std::cout<<"weighed-";
    }
  else
    psnr = pSNRcolor(input1, input2);

  std::cout<<"pSNR = "<<psnr<<" dB"<<std::endl;
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
