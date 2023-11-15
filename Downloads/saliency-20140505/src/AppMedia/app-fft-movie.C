/*!@file AppMedia/app-fft-movie.C Simple application to display the
  frame-by-frame fft magnitude and phase of a movie. */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-fft-movie.C $
// $Id: app-fft-movie.C 5804 2005-10-28 18:18:44Z rjpeters $
//

#ifndef APPMEDIA_APP_FFT_MOVIE_C_DEFINED
#define APPMEDIA_APP_FFT_MOVIE_C_DEFINED

#include "Component/ModelManager.H"
#include "GUI/ImageDisplayStream.H"
#include "Image/ColorOps.H"
#include "Image/Convolver.H"
#include "Image/Coords.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/FourierEngine.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Media/MPEGStream.H"
#include "rutz/trace.h"

int main(int argc, const char** argv)
{
  ModelManager mgr(argv[0]);

  nub::soft_ref<InputMPEGStream> ims(new InputMPEGStream(mgr));

  mgr.addSubComponent(ims);
  mgr.exportOptions(MC_RECURSE);

  if (mgr.parseCommandLine(argc, argv, "infile.mpg", 1, 1) == false)
    return 1;

  mgr.start();

  const std::string infile = mgr.getExtraArg(0);

  ims->setFileName(infile);

  nub::soft_ref<ImageDisplayStream> ids(new ImageDisplayStream(mgr));

  Image<float> box(15, 15, NO_INIT);
  box.clear(1.0f/(15*15));

  LINFO("box sum: %g", sum(box));

  FourierEngine<double> eng(ims->peekDims());
  FourierInvEngine<double> ieng(ims->peekDims());

  Convolver conv(box, ims->peekDims());

  while (true)
    {
      const Image<PixRGB<byte> > img = ims->readRGB();

      if (!img.initialized())
        break;

      const Image<float> lum = luminance(Image<PixRGB<float> >(img));

      Image<complexd> res = eng.fft(Image<double>(lum));

      const Image<double> rt = ieng.ifft(res) / img.getSize();

      const Image<double> logmag = logmagnitude(res);
      const Image<double> phz = phase(res);
      const Image<double> cart = cartesian(logmag, Dims(256, 256));

      ids->writeRGB(img, "input");
      ids->writeFloat(logmag, FLOAT_NORM_0_255 | FLOAT_NORM_WITH_SCALE,
                      "logmag-fft-input");
      ids->writeFloat(phz, FLOAT_NORM_0_255 | FLOAT_NORM_WITH_SCALE,
                      "phase-fft-input");
      ids->writeFloat(cart, FLOAT_NORM_0_255 | FLOAT_NORM_WITH_SCALE,
                      "cartesian-logmag-fft-input");

      const Image<float> c1 = conv.spatialConvolve(lum);

      const Image<float> c2 = conv.fftConvolve(lum);

      ids->writeFloat(c1, FLOAT_NORM_0_255 | FLOAT_NORM_WITH_SCALE,
                      "c1");
      ids->writeFloat(c2, FLOAT_NORM_0_255 | FLOAT_NORM_WITH_SCALE,
                      "c2");

      LINFO("rms conv diff: %e corrcoef: %e",
            RMSerr(c1, c2), corrcoef(c1, c2));

      LINFO("rms roundtrip fft->ifft diff: %e",
            RMSerr(Image<float>(rt), lum));
    }

  mgr.stop();
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_FFT_MOVIE_C_DEFINED
