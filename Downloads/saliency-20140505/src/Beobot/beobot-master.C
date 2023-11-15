/*!@file Beobot/beobot-master.C main control program for beobots */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-master.C $
// $Id: beobot-master.C 9412 2008-03-10 23:10:15Z farhan $
//

/*!
  This parallel vision processing master is for use with beobot-slave.C.
  See the beobotgo script in bin/ for how to launch the slaves.
*/

//#define DEBUG_MODE

#include "Beobot/Beobot.H"
#include "Beobot/beobot-defs.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/RadioDecoder.H"
#include "GUI/XWindow.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/fancynorm.H"
#include "Transport/FrameIstream.H"
#include "Util/Types.H"

#include <cmath>
#include <netdb.h>
#include <signal.h>
#include <unistd.h>

//! Names of the beowulf slaves (default port is 9789)
#define SLAVES "bb1ag:9790 bb1bg:9789 bb1bg:9790"

//! Number of frames over which to compute average stats:
#define NBAVG 10
//! Number of stats to collect:
#define NBSTATS 6

//! define this to use only one cpu
#define USE_SINGLE_CPU 0

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Beobot Main");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<RadioDecoder> radio(new RadioDecoder(manager));
  manager.addSubComponent(radio);

  // choose an IEEE1394grabber by default, and a few custom grabbing
  // defaults, for backward compatibility with an older version of
  // this program:
  manager.setOptionValString(&OPT_FrameGrabberType, "1394");
  manager.setOptionValString(&OPT_FrameGrabberDims, "160x120");

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<hostname> <port>",
                               2, 2) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  int imgwidth = gb->getWidth(), imgheight = gb->getHeight();

  struct hostent *he = gethostbyname(manager.getExtraArg(0).c_str());
  in_addr_t myIP = 0;
  if (he == NULL)
    LFATAL("Cannot determine IP address of %s",manager.getExtraArg(0).c_str());
  else myIP = ntohl( ((in_addr *)(he->h_addr_list[0]))->s_addr );
  short int port = manager.getExtraArgAs<int>(1);
  LINFO("Start on %s[%lx]:%d",
        manager.getExtraArg(0).c_str(), (unsigned long) myIP, port);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // calibrate radio:
  ///radio.zeroCalibrate();
  ///radio.rangeCalibrate();

  // get prepared to grab:
  Image< PixRGB<byte> > *ima; // image to grab
  int frame = 0;              // grabbed frame number

  // initialize beobot:
#if USE_SINGLE_CPU != 0
  Beobot beobot(NULL, imgwidth, imgheight, LEVEL_MIN, LEVEL_MAX,
                DELTA_MIN, DELTA_MAX, SMLEVEL, NBORIENTS, VCXNORM_MAXNORM,
                JETLEVEL, JETDEPTH, NBNEIGH, myIP, port);
#else
  Beobot beobot(SLAVES, imgwidth, imgheight, LEVEL_MIN, LEVEL_MAX,
                DELTA_MIN, DELTA_MAX, SMLEVEL, NBORIENTS, VCXNORM_MAXNORM,
                JETLEVEL, JETDEPTH, NBNEIGH, myIP, port);
#endif

  // get access to beobot's input image so that we can grab it:
  ima = beobot.getRetinaPtr();

  // accumulate processing statistics:
  int stats[NBSTATS]; Timer tim; int navg = 0;
  for (int i = 0; i < NBSTATS; i ++) stats[i] = 0;

#ifdef DEBUG_MODE
  // get some debug displays going:
  XWindow xw(gb->peekDims()), xw2(gb->peekDims());
#endif

  LINFO("All initializations complete. READY.");

  // let's get all our ModelComponent instances started:
  manager.start();

  // ########## MAIN LOOP: grab, process, display:
  while(goforever)
    {
      tim.reset();

      // grab an image:
      *ima = gb->readRGB();
      stats[0] += tim.getReset();

      // start low-level processing on slave nodes:
      if (frame == 0 || USE_SINGLE_CPU) beobot.lowLevel(frame);
      else beobot.lowLevelStart(frame);
      stats[1] += tim.getReset();
      // CAUTION: we do intermediate processing on the low-level results that
      // were collected at the previous frame...

      // intermediate level processing:
      ////      beobot.intermediateLevel(true);
      // always true until we figure a way to reset when messed up

      stats[2] += tim.getReset();

      // high level processing
      ////beobot.highLevel();
      stats[3] += tim.getReset();

#ifdef DEBUG_MODE
      // show the clustered image for debug (before we do decision)
      Image< PixRGB<byte> >tmp;
      ////beobot.DEBUGgetClustered(tmp); xw2.drawImage(tmp);
#endif

      // determine current action
      ////beobot.decision();
      stats[4] += tim.getReset();

      // and finally...
      // we don't move the robot at this point beobot.act();
      BeobotAction a; Point2D<int> win;
      ////beobot.DEBUGgetCurrentAction(a); beobot.getWinner(win);
      float speed = a.getSpeed(), turn = a.getTurn();

      // radio takeover:
      //      float rspeed = radio.getVal(0), rturn = radio.getVal(1);
      //if (fabs(rspeed) > 0.1F) speed = rspeed;
      //if (fabs(rturn) > 0.1F) turn = rturn;

      if (fabs(speed) > 0.2) speed = 0.2;  // be slooooow

      LINFO("frame %d: speed = %.1f, turn = %.1f, winner = [%d, %d]",
            frame, speed, turn, win.i, win.j);

      // override beobot's action, taking into consideration the radio:
      a.setSpeed(speed); a.setTurn(turn); a.setGear(1);
      ////beobot.DEBUGsetCurrentAction(a);

      // actuate:
      ////beobot.action();

      // collect results from low-level processing on slave nodes:
      if (frame > 0 && USE_SINGLE_CPU == 0) beobot.lowLevelEnd(frame);
      stats[5] += tim.getReset();

      // display stats
      navg ++;
      if (navg == NBAVG) {
        LINFO("GB:%.1f LL:%.1f IL:%.1f HL:%.1f DC:%.1f LL:%.1f",
              ((float)(stats[0])) / (float)NBAVG,
              ((float)(stats[1])) / (float)NBAVG,
              ((float)(stats[2])) / (float)NBAVG,
              ((float)(stats[3])) / (float)NBAVG,
              ((float)(stats[4])) / (float)NBAVG,
              ((float)(stats[5])) / (float)NBAVG);
          navg = 0; for (int i = 0; i < NBSTATS; i ++) stats[i] = 0;
      }

#ifdef DEBUG_MODE
      // draw focus of attention on original frame:
      PixRGB<byte> yellow(255, 255, 0);
      ima->drawCircle(win, imgwidth/12, yellow, 2);

      // show the original frame:
      xw.drawImage(*ima);
#endif

      // ready for next frame...
      frame ++;
    }

  // got interrupted; let's cleanup and exit:
  manager.stop();
  exit(0);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
