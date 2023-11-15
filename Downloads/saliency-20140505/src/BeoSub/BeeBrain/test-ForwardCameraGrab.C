/*!@file BeoSub/BeeBrain/test-ForwardCameraGrab.C   grab images from
  forward camera                                                        */
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
// Primary maintainer for this file: Michael Montalbo <montalbo@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/BeoSub/BeeBrain/test-ForwardCameraGrab.C $
// $Id: test-ForwardCameraGrab.C 9601 2008-04-09 23:25:01Z beobot $
//
//////////////////////////////////////////////////////////////////////////

#include "Component/ModelManager.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "GUI/XWinManaged.H"

#include "Media/MediaOpts.H"
#include "Devices/DeviceOpts.H"
#include "Raster/GenericFrame.H"

#include "Image/CutPaste.H"
#include "Image/ShapeOps.H"

#include "Util/Timer.H"
#include <signal.h>

#define NAVG 20

bool goforever = false;

// ######################################################################
//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main( int argc, const char* argv[] )
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("seabee 2007 competition");

  // Instantiate our various ModelComponents:
  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  manager.setOptionValString(&OPT_InputFrameSource, "V4L2");
  manager.setOptionValString(&OPT_FrameGrabberMode, "YUYV");
  manager.setOptionValString(&OPT_FrameGrabberDims, "1024x576");
  manager.setOptionValString(&OPT_FrameGrabberByteSwap, "no");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "30");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  int w = ifs->getWidth(),  h = ifs->getHeight();
  //int w = 320, h = 240;
  std::string dims = convertToString(Dims(w, h));
  LINFO("image size: [%dx%d]", w, h);
  manager.setOptionValString(&OPT_InputFrameDims, dims);

  manager.setModelParamVal("InputFrameDims", Dims(w, h),
                           MC_RECURSE | MC_IGNORE_MISSING);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // let's do it!
  manager.start();

  //start streaming
  ifs->startStream();

  rutz::shared_ptr<XWinManaged> fwin
    (new XWinManaged(Dims(w,h), 20, 20, "Forward Vision Window"));

  Timer tim(1000000); uint64 t[NAVG]; float frate = 0.0f; tim.reset();
  goforever = true;  uint fNum = 0;
  while(goforever)
    {
      // get and store image
      //    ifs->updateNext();

      Image<PixRGB<byte> > ima;// = ifs->readRGB();

      const FrameState is = ifs->updateNext();
      if (is == FRAME_COMPLETE)
        break;

      // grab the images
      GenericFrame input = ifs->readFrame();
      if (!input.initialized()) break;
      ima = rescale(input.asRgb(), 320,240);

      // display the image
      //if((fNum % 15) ==0)
        fwin->drawImage(ima,0,0);

      // compute and show framerate over the last NAVG frames:
      t[fNum % NAVG] = tim.get(); tim.reset();
      if (fNum % NAVG == 0 && fNum > 0)
        {
          uint64 avg = 0ULL;
          for(int i = 0; i < NAVG; i ++) avg += t[i];
          frate = 1000000.0F / float(avg) * float(NAVG);
          LINFO("[%6d] Frame rate: %6.3f fps -> %8.3f ms/frame",
                fNum,frate, 1000.0/frate);
        }

      std::string saveFNameFwd = sformat("cross_FWD_%07d.ppm", fNum);
      LINFO("saving: %s",saveFNameFwd.c_str());
      Raster::WriteRGB(bImg,saveFNameFwd);

      fNum++;
    }

  // we are done
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
