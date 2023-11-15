/*!@file AppPsycho/videograb.C grab frames and save them to disk */

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
// Primary maintainer for this file: Laurent Itti <itti@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppGUI/videograb.C $
// $Id: videograb.C 9816 2008-06-17 01:42:47Z ilab24 $
//

#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Image.H"
#include "Image/ImageCache.H"
#include "Image/Pixels.H"
#include "GUI/SDLdisplay.H"
#include "GUI/GUIOpts.H"
#include "Raster/Raster.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/sformat.H"
#include "Video/RgbConversion.H" // for toVideoYUV422()

#include <pthread.h>

//! number of frames over which frame rate is computed
#define NAVG 20

//! Maximum number of frames in queue
#define MAXQLEN 1000

pthread_mutex_t qmutex;
ImageCache< PixRGB<byte> > cache;
std::vector<std::string> base;

static void* saveframes(void *)
{
  uint fnb = 0;
  while(1) {
    Image< PixRGB<byte> > ima; bool havemore = false;

    // do we have images ready to go?
    pthread_mutex_lock(&qmutex);
    if (cache.size()) ima = cache.pop_front();
    if (cache.size()) havemore = true;
    pthread_mutex_unlock(&qmutex);

    // if we got an image, save it:
    if (ima.initialized())
      {
        // we save each frame to a different base in a rolling manner:
        const char *b = base[fnb % base.size()].c_str();
        Raster::WriteRGB(ima, sformat("%s%06u.ppm", b, fnb++));
      }

    if (havemore == false) usleep(1000);
  }
  return NULL;
}


/*! This simple executable grabs video frames through the video4linux
  driver (see V4Lgrabber.H) or the IEEE1394 (firewire) grabber (see
  IEEE1394grabber.H). Selection of the grabber type is made via the
  --fg-type=XX command-line option. Frames are pushed into a queue and
  a second thread then tries to empty the queue as quickly as possible
  by writing the frames to disk. In testing, PPM format actually gave
  better frame rates than PNG, so that's what is used.  Press <SPACE>
  to start grabbing and <SPACE> again to stop. */
static int submain(const int argc, char** argv)
{
  // instantiate a model manager:
  ModelManager manager("Frame Grabber");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<SDLdisplay> d(new SDLdisplay(manager));
  manager.addSubComponent(d);

  manager.setOptionValString(&OPT_SDLdisplayPriority, "0");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<basename> ... <basename>",
                               1, -1) == false)
    return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");

  // get the basename:
  for (uint i = 0; i < manager.numExtraArgs(); i ++)
    base.push_back(manager.getExtraArg(i));

  // let's get all our ModelComponent instances started:
  manager.start();

  // get ready for main loop:
  Timer tim; uint64 t[NAVG]; int frame = 0;
  d->clearScreen(PixRGB<byte>(128)); bool doit = true;

  int iw = gb->getWidth(), ih = gb->getHeight();
  int dw = d->getDims().w(), dh = d->getDims().h();
  if (iw > dw || ih > dh)
    {
      LERROR("Grab frame size must be smaller than display size");
      doit = false;
    }

  int ovlyoff = (dw - iw) / 2 + dw * ((dh - ih) / 2);
  int ovluvoff = (dw - iw) / 4 + dw * ((dh - ih) / 8);
  int ystride = (dw - iw), uvstride = (dw - iw) / 2;
  bool saving = false; float frate = 0.0f;
  pthread_t saver;
  pthread_create(&saver, NULL, &saveframes, (void *)NULL);

  // create an overlay:
  d->createYUVoverlay(SDL_YV12_OVERLAY);

  // get the frame grabber to start streaming:
  gb->startStream();

  // main loop:
  while(doit) {
    tim.reset();

    // grab a frame:
    Image< PixRGB<byte> > ima = gb->readRGB();

    // to measure display time:
    uint64 t0 = tim.get();

    // if saving, push image into queue:
    if (saving)
      {
        pthread_mutex_lock(&qmutex);
        cache.push_back(ima);
        pthread_mutex_unlock(&qmutex);
        const std::string msg = sformat(" %.1ffps [%04d] ", frate, cache.size());
        writeText(ima, Point2D<int>(0, 0), msg.c_str(), PixRGB<byte>(255), PixRGB<byte>(0));
      }
    else // tell user we are ready to save
      {
        const std::string msg = sformat(" [SPC] to save [%04d] ", cache.size());
        writeText(ima, Point2D<int>(0, 0), msg.c_str(), PixRGB<byte>(255), PixRGB<byte>(0));
      }

    // show the frame:
    SDL_Overlay* ovl = d->lockYUVoverlay();
    toVideoYUV422(ima, ovl->pixels[0] + ovlyoff,
                  ovl->pixels[2] + ovluvoff,
                  ovl->pixels[1] + ovluvoff,
             ystride, uvstride, uvstride);
    d->unlockYUVoverlay();
    d->displayYUVoverlay(-1, SDLdisplay::NO_WAIT);

    // check for space bar pressed; note: will abort us violently if
    // ESC is pressed instead:
    if (d->checkForKey() == ' ') saving = ! saving;

    t[frame % NAVG] = tim.get();
    t0 = t[frame % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);

    // compute and show framerate over the last NAVG frames:
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        frate = 1000.0F / float(avg) * float(NAVG);
      }
    frame ++;
  }

  // stop all our ModelComponents
  d->destroyYUVoverlay();
  manager.stop();

  // all done!
  return 0;
}

extern "C" int main(const int argc, char** argv)
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
