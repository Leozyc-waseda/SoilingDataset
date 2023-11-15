/*!@file CUDA/test-RegSaliency.C tests the reg saliency code */

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
// Primary maintainer for this file
// $HeadURL: svn://dparks@isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/test-RegSaliency.C $
// $Id: test-RegSaliency.C 12962 2010-03-06 02:13:53Z irock $
//


#include "Component/ModelManager.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Image/ShapeOps.H"
#include "Image/Transforms.H"

#include "Neuro/NeuroOpts.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Transport/FrameInfo.H"
#include "Transport/FrameOstream.H"
#include "Util/Timer.H"

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <rutz/rand.h>


// ######################################################################
int main(const int argc, const char **argv)
{

  // instantiate a model manager (for camera input):
  ModelManager manager("Exhaustive video scan");

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<outputdir>", 1, 1) == false) return(1);

  // ######################################################################
  // let's do it!
  manager.start();
  std::string sampleImagesDir;

  sampleImagesDir = manager.getExtraArg(0);


  ifs->startStream();
  int patchSize=72;
  bool quit=false;
  // Randomize the seed
  //rutz::urand r(time((time_t*)0) + getpid());
  // Don't randomize the seed
  int curPickedFrame=0;
  rutz::urand r(0);
  while(!quit)
    {
      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      // Sub sample video, randomly pick roughly every N frames
      int willPick = r.idraw(100);

      if(willPick==0)
        {
          GenericFrame input = ifs->readFrame();
          if (!input.initialized())
            break;

          const Image<PixRGB<byte> > inimg = input.asRgb();
          // Go through image in an exhaustive scan and extract images
          Dims pSize = Dims(patchSize,patchSize);
          int halfSize = patchSize/2;
          int wspan = (inimg.getWidth() / (halfSize*2))*2 - 1;
          int hspan = (inimg.getHeight() / (halfSize*2))*2 - 1;
          for(int i=0;i<wspan&&!quit;i++)
            {
              for(int j=0;j<hspan&&!quit;j++)
                {
                  Image<PixRGB<byte> > patch = crop(inimg,Point2D<int>(i*halfSize,j*halfSize),pSize);
                  Image<PixRGB<byte> > patchResize = rescale(patch, 256, 256);
                  char fname[200];
                  sprintf(fname,"%s/Sample-%d-%d.png",sampleImagesDir.c_str(),curPickedFrame,i*wspan+j);
                  Raster::WriteRGB(patchResize,fname);
                }
            }
          printf("input frame number %d\n",ifs->frame());
          curPickedFrame++;
        }
      // // Just randomly pick a location
      // int ri = r.idraw(inimg.getWidth()-patchSize);
      // int rj = r.idraw(inimg.getHeight()-patchSize);
      // const FrameState os = ofs->updateNext();
      // Image<PixRGB<byte> > out = crop(inimg,Point2D<int>(ri,rj),pSize);
      // //out = rescaleBilinear(out,inimg.getDims());	      
      // GenericFrame output = GenericFrame(out);
      // ofs->writeFrame(output, "test",
      //   	      FrameInfo("exhaustive scan", SRC_POS));
      // if (os == FRAME_FINAL)
      //   {
      //     quit=true;
      //   }

      


      //status = seq->evolve();
    }

  // get ready to terminate:
  manager.stop();

  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
