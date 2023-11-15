/*!@file INVT/neovision2.C CUDA-accelerated Neovision2 integrated demo */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/neovision2-cuda.C $
// $Id: neovision2-cuda.C 13232 2010-04-15 02:15:06Z dparks $
//

#include "Component/ModelManager.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/Layout.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/PyramidOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Transport/TransportOpts.H"
#include "Util/csignals.H"
#include "rutz/shared_ptr.h"
#include "rutz/trace.h"
#include "rutz/rand.h"

#include "CUDA/CudaSaliency.H"

#include <ctype.h>
#include <deque>
#include <iterator>
#include <limits>
#include <stdlib.h> // for atoi(), malloc(), free()
#include <string.h>
#include <sys/resource.h>
#include <signal.h>
#include <time.h>
#include <vector>



// ######################################################################
int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  signal(SIGPIPE, SIG_IGN);
  catchsignals(&signum);

  // Instantiate our various ModelComponents:

  ModelManager manager("Nv2");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<CudaSaliency> cus(new CudaSaliency(manager));
  manager.addSubComponent(cus);

  if (manager.parseCommandLine((const int)argc, (const char**)argv, "<sampleimagesdir>", 1, 1) == false)
    return 1;

  manager.exportOptions(MC_RECURSE);

  manager.start();

  std::string sampleImagesDir;

  sampleImagesDir = manager.getExtraArg(0);


  int retval = 0;
  ifs->startStream();
  const GenericFrameSpec fspec = ifs->peekFrameSpec();
  Image<PixRGB<byte> > rgbin_last;
  // Randomize the seed
  //rutz::urand r(time((time_t*)0) + getpid());
  // Don't randomize the seed, i want the same frames each time
  rutz::urand r(0);

  int curPickedFrame=0;

  while (true)
    {
      if (signum != 0) {
        LINFO("quitting because %s was caught", signame(signum));
        retval = -1;
        break;
      }

      if (ofs->becameVoid()) {
        LINFO("quitting because output stream was closed or became void");
        break;
      }

      //
      // get the next frame from our input source
      //
      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE) break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized()) break;

      const Image<PixRGB<byte> > rgbin = input.asRgb();

      rgbin_last = rgbin;
      cus->doInput(rgbin);


      // Sub sample video, randomly pick roughly every N frames
      int willPick = r.idraw(100);

      if(willPick==0)
        {
            //Different Maps
          CudaImage<float> out = cus->getCudaOutput();

          std::vector<Point2D<int> > loc =  cus->getSalMaxLoc();

          int rectW = cus->getPatchSize();
          int rectH = cus->getPatchSize();
      
          for(unsigned int i=0; i<loc.size(); i++)
            {
              Point2D<int> modLoc;
              modLoc.i = loc[i].i * rgbin.getWidth() / double(out.getWidth());
              modLoc.j = loc[i].j * rgbin.getHeight() / double(out.getHeight());

              int fi = std::max(modLoc.i-int(rectW/2.0),0);
              int fj = std::max(modLoc.j-int(rectH/2.0),0);
              Point2D<int> p = Point2D<int>(fi,fj);
              Rectangle foa = Rectangle(p,Dims(rectW,rectH));
              printf("FOA:<%s>  ",convertToString(modLoc).c_str());
              Image<PixRGB<byte> > patch = crop(rgbin,foa);
              Image<PixRGB<byte> > patchResize = rescale(patch, 256, 256);

              char fname[200];
              sprintf(fname,"%s/Sample-%d-%d.png",sampleImagesDir.c_str(),curPickedFrame,i);
              Raster::WriteRGB(patchResize,fname);

            }
          printf("\n");
          curPickedFrame++;
        }

    }

  manager.stop();
  return retval;
}

// ######################################################################
int main(int argc, const char** argv)
{
  try {
    return submain(argc, argv);
  } catch (...) {
    REPORT_CURRENT_EXCEPTION;
  }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
