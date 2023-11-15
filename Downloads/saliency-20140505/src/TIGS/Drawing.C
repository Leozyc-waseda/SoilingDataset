/*!@file TIGS/Drawing.C */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
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
// Primary maintainer for this file:
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TIGS/Drawing.C $
// $Id: Drawing.C 9412 2008-03-10 23:10:15Z farhan $
//

#ifndef TIGS_DRAWING_C_DEFINED
#define TIGS_DRAWING_C_DEFINED

#include "TIGS/Drawing.H"

#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"

Image<PixRGB<byte> > tigs::boxify(const Image<PixRGB<byte> >& img,
                                  const int border,
                                  const PixRGB<byte>& col)
{
  Image<PixRGB<byte> > result(img.getWidth() + border*2,
                              img.getHeight() + border*2,
                              NO_INIT);

  result.clear(col);

  inplacePaste(result, img, Point2D<int>(border,border));

  return result;
}

Image<PixRGB<byte> > tigs::labelImage(const Image<PixRGB<byte> >& img,
                                      const char* text,
                                      const PixRGB<byte>& col,
                                      const PixRGB<byte>& bg)
{
  Image<PixRGB<byte> > result(img.getWidth(),
                              img.getHeight() + 30, NO_INIT);

  result.clear(bg);

  inplacePaste(result, img, Point2D<int>(0, 30));

  writeText(result, Point2D<int>(0, 8), text, col, bg);

  return result;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TIGS_DRAWING_C_DEFINED
