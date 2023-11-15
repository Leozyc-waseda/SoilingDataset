/*!@file AppMedia/app-thumbnail-movies.C Make a "moving thumbnail" array from a set of movies */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppMedia/app-thumbnail-movies.C $
// $Id: app-thumbnail-movies.C 15310 2012-06-01 02:29:24Z itti $
//

#ifndef APPMEDIA_APP_THUMBNAIL_MOVIES_C_DEFINED
#define APPMEDIA_APP_THUMBNAIL_MOVIES_C_DEFINED

#include "Component/ModelManager.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Media/MpegInputStream.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Util/csignals.H"
#include "Util/sformat.H"
#include "rutz/trace.h"

#include <cmath>
#include <vector>

int submain(int argc, const char** argv)
{
GVX_TRACE(__PRETTY_FUNCTION__);

  volatile int signum = 0;
  catchsignals(&signum);

  ModelManager mgr("movie thumbnailer");

  nub::ref<InputMPEGStream> indummy(new InputMPEGStream(mgr));
  mgr.addSubComponent(indummy);

  nub::ref<OutputFrameSeries> out(new OutputFrameSeries(mgr));
  mgr.addSubComponent(out);

  if (mgr.parseCommandLine(argc, argv, "nskipframes movie1.mpg movie2.mpg ...",
                           2, -1) == false)
    exit(1);

  std::vector<nub::ref<InputMPEGStream> > inputs;

  const int nskipframes = mgr.getExtraArgAs<int>(0);

  LINFO("got nskipframes=%d", nskipframes);

  for (uint i = 1; i < mgr.numExtraArgs(); ++i)
    {
      inputs.push_back(nub::ref<InputMPEGStream>(new InputMPEGStream(mgr)));
      mgr.addSubComponent(inputs.back());
      inputs.back()->setFileName(mgr.getExtraArg(i));
    }

  mgr.exportOptions(MC_RECURSE);
  mgr.start();

  std::vector<Image<PixRGB<byte> > > images(inputs.size());

  const int nx = int(ceil(sqrt(inputs.size())));

  for (int sk = 0; sk < nskipframes; ++sk)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      LINFO("skipping input frame %d", sk);

      bool done = false;

      for (uint i = 0; i < inputs.size(); ++i)
        {
          if (!inputs[i]->readAndDiscardFrame())
            {
              LINFO("quitting because %s reached EOF",
                    mgr.getExtraArg(i).c_str());
              done = true;
            }
        }

      if (done)
        break;
    }

  while (1)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          return -1;
        }

      bool done = false;

      for (uint i = 0; i < inputs.size(); ++i)
        {
          images[i] = inputs[i]->readRGB();
          if (!images[i].initialized())
            {
              LINFO("quitting because %s reached EOF",
                    mgr.getExtraArg(i).c_str());
              done = true;
            }
          else
            images[i] = decXY(lowPass3(images[i]));
        }

      if (done)
        break;

      const Image<PixRGB<byte> > thumb =
        concatArray(&images[0], images.size(), nx);

      const FrameState os = out->updateNext();

      out->writeRGB(thumb, "thumbnail",
                    FrameInfo(sformat("array of %" ZU " movies",
                                      inputs.size()), SRC_POS));

      if (os == FRAME_FINAL)
        {
          LINFO("quitting because output reached EOF");
          break;
        }
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
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */

#endif // APPMEDIA_APP_THUMBNAIL_MOVIES_C_DEFINED
