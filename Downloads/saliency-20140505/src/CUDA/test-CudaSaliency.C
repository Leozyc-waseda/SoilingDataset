/*!@file Demo/test-RunCudaSaliency.C tests the cuda saliency code */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/CUDA/test-CudaSaliency.C $
// $Id: test-CudaSaliency.C 12962 2010-03-06 02:13:53Z irock $
//


#include "Component/ModelManager.H"
#include "CUDA/CudaSaliency.H"
#include "CUDA/CudaPyramidOps.H"
#include "CUDA/CudaShapeOps.H"
#include "CUDA/CudaTransforms.H"
#include "CUDA/CudaPyrBuilder.H"
#include "CUDA/CudaNorm.H"
#include "CUDA/CudaRandom.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ColorOps.H"
#include "Neuro/NeuroOpts.H"
#include "Raster/Raster.H"
#include "Raster/GenericFrame.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Transport/FrameInfo.H"
#include "Transport/FrameOstream.H"
#include "Util/Timer.H"
#include <cuda.h>

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>


// ######################################################################
int main(const int argc, const char **argv)
{

  // instantiate a model manager (for camera input):
  ModelManager manager("CudaSaliency Tester");

  // NOTE: make sure you register your OutputFrameSeries with the
  // manager before you do your InputFrameSeries, to ensure that
  // outputs for the current frame get saved before the next input
  // frame is loaded.
  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  nub::ref<CudaSaliency> smt(new CudaSaliency(manager));
  manager.addSubComponent(smt);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<CudaDev>", 1, 1) == false) return(1);

  std::string devArg;
  devArg = manager.getExtraArg(0);
  MemoryPolicy mp = GLOBAL_DEVICE_MEMORY;
  int dev = strtol(devArg.c_str(),NULL,0);

  smt->setDevice(mp,dev);
  // ######################################################################
  // let's do it!
  manager.start();
  ifs->startStream();
  //unsigned int free,total;

  while(true)
    {
      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      GenericFrame input = ifs->readFrame();
      if (!input.initialized())
        break;

      const Image<PixRGB<byte> > inimg = input.asRgb();
      //cuMemGetInfo(&free, &total); printf("^^^^ CUDA Mem1 -- free %u, total %u\n",free,total);
      smt->doInput(inimg);
      //cuMemGetInfo(&free, &total); printf("^^^^ CUDA Mem2 -- free %u, total %u\n",free,total);

      const FrameState os = ofs->updateNext();
      Image<float> out = smt->getOutput();
      out = rescaleBilinear(out,inimg.getDims());
      const Image<PixRGB<byte> > colout = inimg + toRGB(out);
      GenericFrame output = GenericFrame(colout);

      ofs->writeFrame(output, "test",
                   FrameInfo("cuda saliency frame", SRC_POS));
      printf("input frame number %d\n",ifs->frame());
      if (os == FRAME_FINAL)
        break;
      //status = seq->evolve();
    }

  // get ready to terminate:
  manager.stop();
  cuda_invt_allocation_show_stats(1, "final",0);
  return 0;
}


// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
