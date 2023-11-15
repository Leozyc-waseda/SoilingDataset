/*!@file MBARI/test-BitObject.C test program for the class BitObject
 */
// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2002   //
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
// Primary maintainer for this file: Dirk Walther <walther@caltech.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/MBARI/test-BitObject.C $
// $Id: test-BitObject.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Image/ColorOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "MBARI/BitObject.H"
#include "Raster/Raster.H"
#include "Util/StringConversions.H"
#include "Util/Types.H"
#include "Util/log.H"

int main(const int argc, const char** argv)
{
  if (argc < 2) LFATAL("usage: %s image.ppm [i,j]",argv[0]);

  Image<byte> img = luminance(Raster::ReadRGB(argv[1]));

  LINFO("Image %s loaded.", argv[1]);

  BitObject bobj;

  if (argc >= 3)
    {
      Point2D<int> loc = fromStr<Point2D<int> >(argv[2]);
      LINFO("location = %s",toStr(loc).c_str());
      bobj.reset(img, loc);
      LINFO("BitObject created.");
      if (!bobj.isValid())
        {
          LINFO("No Object found at location %s.",toStr(loc).c_str());
          return 0;
        }

    }
  else
    {
      bobj.reset(img);
      LINFO("BitObject created.");
      if (!bobj.isValid())
        {
          LINFO("No Object found in image %s.",argv[1]);
          return 0;
        }
    }

  LINFO("BoundingBox = %s",toStr(bobj.getBoundingBox()).c_str());
  LINFO("Centroid = %s",toStr(bobj.getCentroid()).c_str());
  LINFO("Area = %d",bobj.getArea());
  float uxx, uyy, uxy;
  bobj.getSecondMoments(uxx,uyy,uxy);
  LINFO("uxx = %g; uyy = %g; uxy = %g",uxx,uyy,uxy);
  LINFO("major Axis = %g", bobj.getMajorAxis());
  LINFO("minor Axis = %g", bobj.getMinorAxis());
  LINFO("elongation = %g", bobj.getElongation());
  LINFO("orientation angle = %g", bobj.getOriAngle());
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
