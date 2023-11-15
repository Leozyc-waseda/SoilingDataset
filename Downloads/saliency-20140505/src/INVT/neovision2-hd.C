/*!@file INVT/neovision2-hd.C Neovision demo for HD streams and 1080p display*/

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/INVT/neovision2-hd.C $
// $Id: neovision2-hd.C 12782 2010-02-05 22:14:30Z irock $
//

#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Component/GlobalOpts.H"
#include "Component/ModelManager.H"
#include "Component/ModelOptionDef.H"
#include "Component/ModelParam.H"
#include "Component/ModelParamBatch.H"
#include "Devices/DeviceOpts.H"
#include "Devices/IEEE1394grabber.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/PrefsWindow.H"
#include "GUI/XWinManaged.H"
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
#include "Media/BufferedInputFrameSeries.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "NeovisionII/Nv2LabelReader.H"
#include "NeovisionII/nv2_common.h"
#include "Neuro/NeoBrain.H"
#include "Neuro/EnvInferoTemporal.H"
#include "Neuro/EnvSaliencyMap.H"
#include "Neuro/EnvSegmenterConfigurator.H"
#include "Neuro/EnvVisualCortex.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Transport/FrameInfo.H"
#include "Transport/TransportOpts.H"
#include "Util/FpsTimer.H"
#include "Util/Pause.H"
#include "Util/StringConversions.H"
#include "Util/StringUtil.H"
#include "Util/SyncJobServer.H"
#include "Util/SysInfo.H"
#include "Util/TextLog.H"
#include "Util/WorkThreadServer.H"
#include "Util/csignals.H"
#include "rutz/shared_ptr.h"
#include "rutz/trace.h"

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

#define PREFERRED_TEXT_LENGTH 42

struct Nv2UiData
{
  Nv2UiData(const int map_zoom_) :
    map_zoom(map_zoom_), ncpu(numCpus())
  {}

  FpsTimer::State time_state;
  const int map_zoom;
  const int ncpu;
};

std::string time2str(const rutz::time& t)
{
  const double s = t.sec();

  const int hpart = int(s/(60.0*60.0));
  const int mpart = int((s-hpart*60*60)/60.0);
  const double spart = s - hpart*60*60 - mpart*60;

  return sformat("%02d:%02d:%06.3f", hpart, mpart, spart);
}

class Nv2UiJob : public JobServer::Job
{
public:
  Nv2UiJob(OutputFrameSeries* ofs_, const Nv2UiData& uidata_,
           Image<PixRGB<byte> > rgbin_, Image<byte> vcxmap_,
           Image<byte> Imap_, Image<byte> Cmap_, Image<byte> Omap_
#ifdef ENV_WITH_DYNAMIC_CHANNELS
           , Image<byte> Fmap_, Image<byte> Mmap_
#endif
                ) :
    ofs(ofs_), uidata(uidata_), rgbin(rgbin_), vcxmap(vcxmap_),
    Imap(Imap_), Cmap(Cmap_), Omap(Omap_)
#ifdef ENV_WITH_DYNAMIC_CHANNELS
    , Fmap(Fmap_), Mmap(Mmap_)
#endif
  { }

  virtual void run()
  {
    ofs->updateNext();

    Point2D<int> win; byte winval;
    findMax(vcxmap, win, winval);
    win = win * uidata.map_zoom;

    // do a markup on the input images:
    Image<PixRGB<byte> > markup = rgbin;
    drawCircle(markup, win, 3, PixRGB<byte>(60, 220, 255), 3);

    // do the saliency map:
    Image<byte> sm = vcxmap; // for now
    Image<PixRGB<byte> > smc =
      zoomXY(sm, uidata.map_zoom / 2, uidata.map_zoom / 2);
    drawCircle(smc, win/2, 3, PixRGB<byte>(60, 220, 255), 3);

    const std::string lines[2] =
      {
        sformat("peak %3d @ (%3d,%3d)",
                winval, win.i, win.j),
        sformat("%s [%5.2ffps, %5.1f%%CPU]",
                time2str(uidata.time_state.elapsed_time).c_str(),
                uidata.time_state.recent_fps,
                uidata.time_state.recent_cpu_usage*100.0)
      };

    const Image<PixRGB<byte> > textarea =
      makeMultilineTextBox(smc.getWidth(), &lines[0], 2,
                           PixRGB<byte>(255, 255, 0), PixRGB<byte>(0,0,0),
                           PREFERRED_TEXT_LENGTH);

    Layout<PixRGB<byte> > disp = vcat(smc, textarea);

    //   ofs->writeRgbLayout(disp, "neovision2-HD Control",
    //                   FrameInfo("copy of input", SRC_POS));
    ofs->writeRGB(markup, "neovision2-HD",
                  FrameInfo("copy of input", SRC_POS));
  }

  virtual const char* jobType() const { return "Nv2UiJob"; }

private:
  OutputFrameSeries* const ofs;
  const Nv2UiData uidata;
  Image<PixRGB<byte> > rgbin;
  const Image<byte> vcxmap;
  const Image<byte> Imap;
  const Image<byte> Cmap;
  const Image<byte> Omap;
#ifdef ENV_WITH_DYNAMIC_CHANNELS
  const Image<byte> Fmap;
  const Image<byte> Mmap;
#endif
};

// ######################################################################
int submain(int argc, const char** argv)
{
  volatile int signum = 0;
  signal(SIGPIPE, SIG_IGN);
  catchsignals(&signum);

  // Instantiate our various ModelComponents:

  ModelManager manager("Nv2hd");

  nub::ref<BufferedInputFrameSeries>
    ifs(new BufferedInputFrameSeries(manager, 50, true));
  manager.addSubComponent(ifs);

  nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
  manager.addSubComponent(ofs);

  nub::ref<EnvVisualCortex> evc(new EnvVisualCortex(manager));
  manager.addSubComponent(evc);

  manager.exportOptions(MC_RECURSE);

#if defined(HAVE_IEEE1394)
  // input comes from firewire camera 640x480/rgb/30fps by default
  manager.setOptionValString(&OPT_InputFrameSource, "ieee1394");
  manager.setOptionValString(&OPT_FrameGrabberMode, "RGB24");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "30");
#elif defined(HAVE_QUICKTIME_QUICKTIME_H)
  manager.setOptionValString(&OPT_InputFrameSource, "qtgrab");
  manager.setOptionValString(&OPT_FrameGrabberDims, "640x480");
#endif

  // output goes to the screen by default
  manager.setOptionValString(&OPT_OutputFrameSink, "display");

  // change some default values
  manager.setOptionValString(&OPT_EvcColorSmoothing, "true");

  if (manager.parseCommandLine(argc, argv,
                               "<ip1:port1,ip2:port2,...>",
                               0, 1) == false)
    return(1);

  manager.start();

  Nv2UiData uidata(1 << evc->getMapLevel());

  // set up a background job server with one worker thread to
  // handle the ui jobs:
  rutz::shared_ptr<WorkThreadServer> tsrv
    (new WorkThreadServer("neovision2-ui", 1));

  // keep max latency low, and if we get bogged down, then drop
  // old frames rather than new ones
  tsrv->setMaxQueueSize(3);
  tsrv->setDropPolicy(WorkThreadServer::DROP_OLDEST);
  tsrv->setFlushBeforeStopping(true);
  rutz::shared_ptr<JobServer> uiq = tsrv;

  /////////////  ifs->startStream();

  const GenericFrameSpec fspec = ifs->peekFrameSpec();
  Image<PixRGB<byte> > rgbin_last(fspec.dims, ZEROS);
  FpsTimer fps_timer;

  int retval = 0; bool toggle = true; Image<byte> vcxmap;
  while (true)
    {
      if (signum != 0)
        {
          LINFO("quitting because %s was caught", signame(signum));
          retval = -1;
          break;
        }

      if (ofs->becameVoid())
        {
          LINFO("quitting because output stream was closed or became void");
          break;
        }

      //
      // get the next frame from our input source
      //
      bool underflowed;
      Image<PixRGB<byte> > rgbin;
      GenericFrame input = ifs->get(&underflowed);
      if (!input.initialized())
        {
          if (underflowed) { LINFO("Input underflow!"); rgbin = rgbin_last; }
          else break; // end of stream or error
        }
      else
        rgbin = input.asRgb();

      rgbin_last = rgbin;

      //
      // send the frame to the EnvVisualCortex and get the vcx output
      //
      if (toggle)
        {
          evc->input(rgbin);
          vcxmap = evc->getVCXmap();
        }

      toggle = !toggle;

      fps_timer.nextFrame();
      uidata.time_state = fps_timer.getState();

      if (uidata.time_state.frame_number % 50 == 0)
        LINFO("frame %u: %.2f fps",
              uidata.time_state.frame_number,
              uidata.time_state.recent_fps);

      //
      // build a ui job to run in the background to display update the
      // saliency map the input frame, the vcx maps,
      //
      uiq->enqueueJob(rutz::make_shared
                      (new Nv2UiJob
                       (ofs.get(), uidata, rgbin, vcxmap, evc->getImap(),
                        evc->getCmap(), evc->getOmap()
#ifdef ENV_WITH_DYNAMIC_CHANNELS
                        , evc->getFmap(), evc->getMmap()
#endif
                        )));
    }

  // destroy the ui queue so that we force it to shut down now
  uiq.reset(0);

  manager.stop();

  return retval;
}

// ######################################################################
int main(int argc, const char** argv)
{
  try
    {
      return submain(argc, argv);
    }
  catch (...)
    {
      REPORT_CURRENT_EXCEPTION;
    }
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* mode: c++ */
/* indent-tabs-mode: nil */
/* End: */
