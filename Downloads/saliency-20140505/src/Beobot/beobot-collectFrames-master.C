/*!@file Beobot/beobot-collectFrames-master.C
  Run beobot-collectFrames-master at A to collect frames
  Run beobot-collectFrames at B to remote control motion */

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
// Primary maintainer for this file: Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-collectFrames-master.C $
// $Id: beobot-collectFrames-master.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Util/Types.H"
#include "Util/log.H"

#include <cstdio>
#include <cstdlib>
#include <signal.h>
#include <unistd.h>

#include "Devices/FrameGrabberConfigurator.H"
#include "Image/DrawOps.H"
#include "Transport/FrameIstream.H"
//#include "Image/Image.H"
#include "Image/ImageCache.H"
#include "Image/Pixels.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
// #include <pthread.h>

#include "GUI/XWinManaged.H"
#include "GUI/XWindow.H"

//! number of frames over which frame rate is computed
#define NAVG 20

//! Maximum number of frames in queue
#define MAXQLEN 1000


pthread_mutex_t qmutex;
pthread_mutex_t smutex;
ImageCache< PixRGB<byte> > cache;
std::string base;
int sessionNum = 0;

static bool goforever = true;

// ######################################################################
void* saveframes(void *)
{
  int fnb = 0; int sNum= -1;
  while(1) {
    Image< PixRGB<byte> > ima; bool havemore = false;

    // do we have images ready to go?
    pthread_mutex_lock(&qmutex);
    if (cache.size()) ima = cache.pop_front();
    if (cache.size()) havemore = true;
    pthread_mutex_unlock(&qmutex);

    int tNum;
    pthread_mutex_lock(&smutex);
    tNum = sessionNum-1;
    pthread_mutex_unlock(&smutex);

    if(sNum < tNum)
      {
        sNum = tNum;
        fnb = 0;
      }

    // if we got an image, save it:
    if (ima.initialized())
      {
        Raster::WriteRGB(ima, sformat("%s%03d_%06d.ppm",
                                      base.c_str(), sNum, fnb));
        LINFO("saving %s%03d_%06d", base.c_str(), sNum, fnb);
        fnb++;
      }
    if (havemore == false) usleep(1000);
  }
  return NULL;
}

// ######################################################################
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
/*! Displays and grabs video feeds to disk.  Selection of  grabber type
    is made via the --fg-type=XX command-line option.
    Frames are pushed into a queue and a second thread then
    tries to empty the queue as quickly as possible by writing the
    frames to disk. In testing, PPM format actually gave better frame
    rates than PNG, which is what is used.

    The starting command is supplied from the beoChip keyboard
*/
int main(const int argc, const char **argv)
{
  // instantiate a model manager:
  ModelManager manager("Beobot: collect frames - Master");

  // Instantiate our various ModelComponents:
  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<base>", 1, 1) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
            "--fg-type=XX command-line option for this program "
           "to be useful -- ABORT");
  //int w = gb->getWidth(), h = gb->getHeight();

  // display window
  XWindow win(gb->peekDims(), -1, -1, "grab window");

  TCPmessage rmsg;      // buffer to receive messages from nodes
  TCPmessage smsg;      // buffer to send messages to nodes

  // get the basename:
  base = manager.getExtraArg(0);//"../data/beobotCF_";

  // let's get all our ModelComponent instances started:
  manager.start();

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get ready for main loop:
  int32 rframe = 0, raction = 0, rnode = 0;
  // this code is just for one slave for now
  //const int nbnode = beo->getNbSlaves(); // number of slave nodes

  // activate the saving thread
  Timer tim; uint64 t[NAVG]; float frate = 0.0f;
  pthread_t saver;
  pthread_create(&saver, NULL, &saveframes, (void *)NULL);
  bool saving = false;

  // get the frame grabber to start streaming:
  gb->startStream();

  // send image to slave node to initialize contact:
  smsg.reset(rframe, 1);
  smsg.addFloat(2.3434);
  beo->send(rnode, smsg);

  // grab, display and save:
  int fNum = 0;
  while(goforever)
  {
    tim.reset();

    // grab a frame
    Image< PixRGB<byte> > ima = gb->readRGB();

    // to measure display time:
    uint64 t0 = tim.get();

    // check if we need to start(1) or stop(0) saving
    if(beo->receive(rnode, rmsg, rframe, raction, 5))
      {
        LINFO("receiving");
        const int32 val = rmsg.getElementInt32();
        bool msg = bool(val);
        rmsg.reset(rframe, raction);
        if(msg)
          LINFO("start capturing");
        else
          LINFO("stop capturing");

        // if we are start saving
        if(!saving & msg)
          {
            pthread_mutex_lock(&smutex);
            sessionNum++;
            pthread_mutex_unlock(&smutex);
            saving = true;
          }
        else if(saving & !msg)
          {
            saving = false;
          }
      }

    // if saving, push image into queue:
    if (saving)
      {
        pthread_mutex_lock(&qmutex);
        cache.push_back(ima);
        pthread_mutex_unlock(&qmutex);
        char msg[30]; sprintf(msg, " %.1ffps [%04d] ", frate, cache.size());
        writeText(ima,Point2D<int>(0, 0), msg, PixRGB<byte>(255), PixRGB<byte>(0));
      }
    else // tell user we are ready to save
      {
        char msg[30]; sprintf(msg, " [SPC] to save [%04d] ", cache.size());
        writeText(ima,Point2D<int>(0, 0), msg, PixRGB<byte>(255), PixRGB<byte>(0));
      }

    // show the frame:
    win.drawImage(ima);

    t[fNum % NAVG] = tim.get();
    t0 = t[fNum % NAVG] - t0;
    if (t0 > 20000ULL) LINFO("Display took %lluus", t0);

    // compute and show framerate over the last NAVG frames:
    if (fNum % NAVG == 0 && fNum > 0)
      {
        uint64 avg = 0ULL; for (int i = 0; i < NAVG; i ++) avg += t[i];
        frate = 1000.0F / float(avg) * float(NAVG);
      }

      fNum++;
    }

  // stop all our ModelComponents
  manager.stop();

  // all done!
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
