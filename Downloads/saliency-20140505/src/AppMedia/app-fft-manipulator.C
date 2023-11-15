/*!@file AppMedia/app-fft-manipulator.C Manipulate an image in the fft
   domain. Used to remove phase or magnitude from an image or movie */

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
// Primary maintainer for this file: David J. Berg <dberg@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-fft-manipulator.C $

#ifndef APPMEDIA_APP_FFT_MANIPULATOR_C_DEFINED
#define APPMEDIA_APP_FFT_MANIPULATOR_C_DEFINED

#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"

#include "Image/ColorOps.H"
#include "Image/FourierEngine.H"
#include "Image/Image.H"
#include "Image/Normalize.H"
#include "Image/MathOps.H"
#include "Media/FrameSeries.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Util/Pause.H"
#include "Util/csignals.H"
#include "Util/Timer.H"

#include <complex>


static const ModelOptionCateg MOC_FFTMANIP = {
  MOC_SORTPRI_2, "Options for fft manipulation" };

static const ModelOptionDef OPT_RemovePhase =
  { MODOPT_FLAG, "RemovePhase", &MOC_FFTMANIP, OPTEXP_CORE,
    "remove the phase component of an image",
    "remove-phase", '\0', "--[no]remove-phase", "false" };

static const ModelOptionDef OPT_RemoveMagnitude =
  { MODOPT_FLAG, "RemoveMagnitude", &MOC_FFTMANIP, OPTEXP_CORE,
    "remove the phase component of an image",
    "remove-magnitude", '\0', "--[no]remove-phase", "false" };

static const ModelOptionDef OPT_AdjustMagnitude =
  { MODOPT_ARG(float), "AdjustMagnitude", &MOC_FFTMANIP, OPTEXP_CORE,
    "adjust the power spectrum of an image by 1/frequency^a",
    "adjust-magnitude", '\0', "<float>", "0.0"  };


double compute_factor(const complexd& val, const double& mag) 
{
  return  mag*mag / sqrt(val.real()*val.real() + val.imag()*val.imag());
}

int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager manager("FFT-manipulator");

  OModelParam<bool> remPhase(&OPT_RemovePhase, &manager);
  OModelParam<bool> remMag(&OPT_RemoveMagnitude, &manager);
  OModelParam<float> adjMag(&OPT_AdjustMagnitude, &manager);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::soft_ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false)
      return(1);

  manager.start();

  ifs->startStream();

  int c = 0;

  PauseWaiter p;

  SimTime tm = SimTime::ZERO();

  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          return 0;
        }

      if (p.checkPause())
        continue;

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;

      const Image<PixRGB<byte> > rgbin = input.asRgb();
      const Image<float> lin = luminance(rgbin);
      const Image<double> lind = lin;
      
      const FrameState os = ofs->updateNext();
      
      //manipulation here
      FourierEngine<double> itsTransform(lind.getDims());
      Image<complexd> fimage = itsTransform.fft(lind);
      
      if (remPhase.getVal())
        {
          Image<complexd>::iterator i = fimage.beginw();
          while (i != fimage.endw())
            {
              const double mag = abs(*i);
              const complexd conj(i->real(), -1 * i->imag());
              const complexd temp = *i * conj;
              const double fac = compute_factor(temp, mag);
              *i++ = complexd(fac * temp.real(), fac * temp.imag());
            }
        }
      
      if (remMag.getVal())
        {
          Image<complexd>::iterator i = fimage.beginw();
          while (i != fimage.endw())
            {
              const double fac = compute_factor(*i, 1.0);
              *i = complexd(fac * i->real(), fac * i->imag());
              ++i;
            }
        }
      
      if(adjMag.getVal() > 0.0F)
        {
          const float exp = -1.0 * adjMag.getVal() / 2.0;
          
          Image<double> mask(fimage.getDims(), NO_INIT);
          Image<double>::iterator i = mask.beginw();
          for (int v = 0; v < fimage.getHeight(); ++v)
            for (int u = 0; u < fimage.getWidth(); ++u)
              if ((v == 0) && (u == 0))
                *i++ = 1.0;
              else
                *i++ = pow(u*u + v*v, exp);  
          
          Image<complexd>::iterator f = fimage.beginw();
          Image<double>::const_iterator m = mask.begin();
          while (f != fimage.endw())
            {
              const double fac = compute_factor(*f, *m);
              *f = complexd(fac * f->real(), fac * f->imag());
              ++f; ++m;
            }               
        }
     
      FourierInvEngine<double> itsInvTransform(lind.getDims());
      const Image<double> ifimage = itsInvTransform.ifft(fimage);
      Image<double> out = ifimage / ifimage.getSize();

      double mn,mx;
      getMinMax(out,mn,mx);
      out = out - mn;
      getMinMax(out,mn,mx);
      out /= mx;
      out *= 255.0;


      const Image<float> outf = out;
      ofs->writeFloat(outf, FLOAT_NORM_PRESERVE, "output");
      
      if (os == FRAME_FINAL)
        break;

      LDEBUG("frame %d", c++);

      if (ifs->shouldWait() || ofs->shouldWait())
        Raster::waitForKey();

      tm += SimTime::HERTZ(30);
    }

  return 0;
}

int main(const int argc, const char **argv)
{
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }

  return 1;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_FFT_MANIPULATOR_C_DEFINED
