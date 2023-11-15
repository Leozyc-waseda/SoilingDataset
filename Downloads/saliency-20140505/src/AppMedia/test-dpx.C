/*!@file AppMedia/test-dpx.C */

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
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/test-dpx.C $
// $Id: test-dpx.C 12074 2009-11-24 07:51:51Z itti $
//

#ifndef APPMEDIA_TEST_DPX_C_DEFINED
#define APPMEDIA_TEST_DPX_C_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Raster/DpxParser.H"
#include "Raster/GenericFrame.H"
#include "Util/StringConversions.H"
#include "Util/sformat.H"

#include <cstdio>

int main(int argc, char** argv)
{
  MYLOGVERB = LOG_DEBUG;

  float gamma = 0.6;
  bool do_log = false;
  float sigm_contrast = 10.0;
  float sigm_thresh = 0.1;
  float sclip_lo = 0.0f;
  float sclip_hi = 5351.0f;
  int reduce = 0;

  if (argc < 2 || argc > 9)
    {
      fprintf(stderr, "usage: %s image.dpx [gamma=%g] [do_log=%d] "
              "[sigm_contrast=%g] [sigm_thresh=%g] "
              "[sclip_lo=%g] [sclip_hi=%g] [reduce=%d]\n",
              argv[0], gamma, int(do_log),
              sigm_contrast, sigm_thresh,
              sclip_lo, sclip_hi, reduce);
      return -1;
    }

  if (argc >= 3) gamma = fromStr<float>(argv[2]);
  if (argc >= 4) do_log = fromStr<bool>(argv[3]);
  if (argc >= 5) sigm_contrast = fromStr<float>(argv[4]);
  if (argc >= 6) sigm_thresh = fromStr<float>(argv[5]);
  if (argc >= 7) sclip_lo = fromStr<float>(argv[6]);
  if (argc >= 8) sclip_hi = fromStr<float>(argv[7]);
  if (argc >= 9) reduce = fromStr<int>(argv[8]);

  const char* fname = argv[1];

  DpxParser dpx(fname,
                gamma, do_log,
                sigm_contrast, sigm_thresh,
                sclip_lo, sclip_hi);

  Image<PixRGB<float> > cimg = dpx.getFrame().asRgbF32();

  for (int i = 0; i < reduce; ++i)
    cimg = quickLocalAvg2x2(cimg);

  XWinManaged xwin(Image<PixRGB<byte> >(cimg),
                   sformat("%s (gamma=%f, sigc=%f, sigt=%f)",
                           fname, gamma, sigm_contrast, sigm_thresh).c_str());

  while (!xwin.pressedCloseButton())
    sleep(1);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_TEST_DPX_C_DEFINED
