/*!@file TestSuite/test-retinex.C */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/TestSuite/test-retinex.C $
// $Id: test-retinex.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef TESTSUITE_TEST_RETINEX_C_DEFINED
#define TESTSUITE_TEST_RETINEX_C_DEFINED

#include "GUI/XWinManaged.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/Range.H"
#include "Image/Retinex.H"
#include "Raster/PfmWriter.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Assert.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "rutz/time.h"

#include <fstream>
#include <string>
#include <unistd.h>

int main(int argc, char** argv)
{
  MYLOGVERB=LOG_INFO;

  if (argc < 2 || argc > 5)
    {
      LERROR("usage: %s imagefile ?do-xwindows? ?do-save? ?gamma?", argv[0]);
      return -1;
    }

  const char* const fname = argv[1];
  const bool interactive = argc > 2 && atoi(argv[2]) != 0;
  const bool dosave = argc > 3 && atoi(argv[3]) != 0;
  const float gam = argc > 4 ? atof(argv[4]) : 0.5;

  const Image<PixRGB<float> > rgbf = Raster::ReadFrame(fname).asRgbF32();
  Image<float> r, g, b;
  getComponents(rgbf, r, g, b);

  XWinManaged* xwin = 0;
  if (interactive)
    {
      xwin = new XWinManaged(Image<PixRGB<byte> >(rgbf), "rgb");

      Image<float> r2 = toPower(r, gam);
      Image<float> g2 = toPower(g, gam);
      Image<float> b2 = toPower(b, gam);

      inplaceNormalize(r2, 0.0f, 255.0f);
      inplaceNormalize(g2, 0.0f, 255.0f);
      inplaceNormalize(b2, 0.0f, 255.0f);

      new XWinManaged(makeRGB(Image<byte>(r2),
                              Image<byte>(g2),
                              Image<byte>(b2)),
                      "rgb^gamma");

      new XWinManaged(Image<byte>(r2), "r^gamma");
      new XWinManaged(Image<byte>(g2), "g^gamma");
      new XWinManaged(Image<byte>(b2), "b^gamma");
    }

  const int niter = 4;

  const size_t depth = retinexDepth(rgbf.getDims());

  LINFO("depth = %" ZU , depth);

  rutz::time t1 = rutz::time::rusage_now();

//   const Rectangle outrect(Point2D<int>(128,128), Dims(128,128));
  const Rectangle outrect(Point2D<int>(0,0), rgbf.getDims());

  const ImageSet<float> RR =
    buildPyrRetinexLog<float>(log(r * 256.0F + 1.0F), depth, niter, outrect);
  const ImageSet<float> GG =
    buildPyrRetinexLog<float>(log(g * 256.0F + 1.0F), depth, niter, outrect);
  const ImageSet<float> BB =
    buildPyrRetinexLog<float>(log(b * 256.0F + 1.0F), depth, niter, outrect);

  rutz::time t2 = rutz::time::rusage_now();

  LINFO("pyramid time = %.4fs", (t2 - t1).sec());

  for (size_t i = 0; i < RR.size(); ++i)
    {
      if (dosave)
        {
          PfmWriter::writeFloat(Image<float>(RR[i]),
                                sformat("rlev%" ZU ".pfm", i));
          PfmWriter::writeFloat(Image<float>(GG[i]),
                                sformat("glev%" ZU ".pfm", i));
          PfmWriter::writeFloat(Image<float>(BB[i]),
                                sformat("blev%" ZU ".pfm", i));
        }

      if (i == 0)
        {
          Range<float> rrng = rangeOf(RR[i]);
          Range<float> grng = rangeOf(GG[i]);
          Range<float> brng = rangeOf(BB[i]);

          LINFO("rrng = %f .. %f", rrng.min(), rrng.max());
          LINFO("grng = %f .. %f", grng.min(), grng.max());
          LINFO("brng = %f .. %f", brng.min(), brng.max());

          Image<float> R = exp(RR[i]*gam);
          Image<float> G = exp(GG[i]*gam);
          Image<float> B = exp(BB[i]*gam);

          inplaceNormalize(R, 0.0f, 255.0f);
          inplaceNormalize(G, 0.0f, 255.0f);
          inplaceNormalize(B, 0.0f, 255.0f);

          if (interactive)
            {
              const Image<PixRGB<byte> > outrgb =
                makeRGB(Image<byte>(R), Image<byte>(G), Image<byte>(B));

              const Image<PixRGB<byte> > outrgbcrop = crop(outrgb, outrect);

              new XWinManaged(outrgb, sformat("Retinex[%" ZU "]", i).c_str());
              new XWinManaged(outrgbcrop, sformat("cropped Retinex[%" ZU "]", i).c_str());
              new XWinManaged(Image<byte>(R), sformat("R[%" ZU "]", i).c_str());
              new XWinManaged(Image<byte>(G), sformat("G[%" ZU "]", i).c_str());
              new XWinManaged(Image<byte>(B), sformat("B[%" ZU "]", i).c_str());
            }
        }
    }

  while (xwin && !xwin->pressedCloseButton())
    {
      usleep(20000);
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // TESTSUITE_TEST_RETINEX_C_DEFINED
