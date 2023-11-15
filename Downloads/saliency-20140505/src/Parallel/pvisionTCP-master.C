/*!@file Parallel/pvisionTCP-master.C Grab video & process over beowulf w/ pvisionTCP */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/pvisionTCP-master.C $
// $Id: pvisionTCP-master.C 9412 2008-03-10 23:10:15Z farhan $
//

/* This parallel vision processing master is for use with pvisionTCP.
  See the pvisionTCPgo script in bin/ for how to launch the slaves.
*/

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "Parallel/pvisionTCP-defs.H"
#include "Transport/FrameIstream.H"
#include "Util/Assert.H"

#include <signal.h>


//! Number of frames over which average framerate is computed
#define NAVG 20
//! Number of stats collected
#define NSTAT 6

//! prescale level by which we downsize images before sending them off
#define PRESCALE 2

static bool goforever = true;  //!< Will turn false on interrupt signal

//! Signal handler (e.g., for control-C)
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Parallel Vision TCP Version 2");

  // Instantiate our various ModelComponents:
  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

  nub::soft_ref<Beowulf>
    beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
  manager.addSubComponent(beo);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "<nframes> <framepause>", 2, 2)
      == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  int w = gb->getWidth(), h = gb->getHeight();

  int nframes = manager.getExtraArgAs<int>(0);
  if (nframes < 1 || nframes > 2048) LFATAL("Invalid number of frames");
  uint64 framepause = manager.getExtraArgAs<uint64>(1);
  if (framepause < 1 || framepause > 2048) LFATAL("Invalid framepause");

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get prepared to grab, communicate, display, etc:
  XWindow xw(gb->peekDims(), -1, -1, "USC Saliency Cam");
  int32 frame = 0;                // count the frames
  PixRGB<byte> pix(255, 255, 0);  // yellow color for fixations
  Point2D<int> win(w/2, h/2); // coordinates of attended location
  int foa_size = std::min(w, h) / 12; // radius of focus of attention
  TCPmessage rmsg;      // buffer to receive messages from nodes
  TCPmessage smsg;      // buffer to send messages to nodes
  Timer tim;            // for computation of framerate
  int t[NSTAT][NAVG], ff; // to compute average framerate and other stats
  int latency = 0, nlat = 0;  // to compute average processing latency
  int dropped = 0;    // number of frames dropped lately

  // setup array of frames:
  Image< PixRGB<byte> > ima[nframes];

  // let's get all our ModelComponent instances started:
  manager.start();

  // get the frame grabber to start streaming:
  gb->startStream();

  // ########## MAIN LOOP: grab, process, display:
  while(goforever)
    {
      // initialize the timer:
      tim.reset(); ff = frame % NAVG;

      // grab an image:
      int32 index = frame % nframes;
      ima[index] = gb->readRGB();
      t[0][ff] = tim.get();  // grab time

      // display current processed frame, i.e., grabbed frame-nframes+1 ago:
      if (frame >= nframes)
        xw.drawImage(ima[(frame - nframes + 1) % nframes]);
      t[1][ff] = tim.get() - t[0][ff];  // display time
      if (t[1][ff] > 20)
        LINFO("*** Display took %dms for frame %d", t[1][ff], frame);

      // send every image to processing, unless we are recovering from drops:
      if (dropped == 0)
        {
          // prescale image:
          Image< PixRGB<byte> > tmpi = ima[index];
          tmpi = rescale(tmpi,
                         tmpi.getWidth() >> PRESCALE,
                         tmpi.getHeight() >> PRESCALE);

          // select processing branch based on frame number:
          int32 offset; if (frame & 1) offset = POFFSET; else offset = 0;

          // we do the job of BEO_RETINA on the master to reduce latency:
          // compute luminance and send it off:
          Image<byte> lum = luminance(tmpi);
          smsg.reset(frame, BEO_LUMINANCE);
          smsg.addImage(lum);
          beo->send(offset + 0, smsg);  // send off to luminance slave

          // compute RG and BY and send them off:
          Image<byte> r, g, b, y; getRGBY(tmpi, r, g, b, y, (byte)25);
          smsg.reset(frame, BEO_REDGREEN);
          smsg.addImage(r); smsg.addImage(g);
          beo->send(offset + 1, smsg);  // send off to RG slave
          smsg.reset(frame, BEO_BLUEYELLOW);
          smsg.addImage(b); smsg.addImage(y);
          beo->send(offset + 2, smsg);  // send off to BY slave

          //LINFO("sent frame %d", frame);
        }
      t[2][ff] = tim.get() - t[1][ff] - t[0][ff];  // send out time

      // receive current coordinates of focus of attention:
      int32 rframe, raction, rnode = -1, recnb = 0;  // receive from any node
      while(beo->receive(rnode, rmsg, rframe, raction, 5)) // wait up to 5ms
        {
          // accumulate data for average latency computation:
          latency += frame - rframe; nlat ++;
          //LINFO("received %d/%d from %d while at %d",
          //    rframe, raction, rnode, frame);
          switch(raction)
            {
            case BEO_WINNER: // ##############################
              {
                const Fixation fix = rmsg.getElementFixation();
                win.i = fix.i; win.j = fix.j; rframe = fix.frame;

                // are we going to keep or drop this result?
                if (rframe > frame - nframes)
                  {

                    // rescale winner coordinates according to PRESCALE:
                    win.i <<= PRESCALE;
                    win.i += int(((1 << (PRESCALE - 1)) *
                                  float(rand()) ) / RAND_MAX);
                    win.j <<= PRESCALE;
                    win.j += int(((1 << (PRESCALE - 1)) *
                                  float(rand()) ) / RAND_MAX);

                    // plot focus of attention onto image, at winning coords:
                    int32 fidx = rframe % nframes;
                    drawPatch(ima[fidx], win, 3, pix);
                    drawCircle(ima[fidx], win, foa_size, pix, 3);

                    if (dropped) dropped --;
                  }
                else
                  {
                    dropped += 3;
                    LINFO("    dropping frame %d (%d) [now %d]; dropped = %d",
                          rframe, rframe % nframes, frame, dropped);
                  }
              }
              break;
            default: // ##############################
              LERROR("Bogus action %d -- IGNORING.", raction);
              break;
            }
          // limit number of receives, so we don't hold CPU too long:
          recnb ++; if (recnb > 4) break;
        }
      t[3][ff] = tim.get() - t[2][ff] - t[1][ff] - t[0][ff]; // receive time

      // compute and show framerate and stats over the last NAVG frames:
      if (ff == 1 && frame > 1)
        {
          int avg[NSTAT];
          for (int j = 0; j < NSTAT; j ++)
            {
              avg[j] = 0;
              for (int i = 0; i < NAVG; i ++) avg[j] += t[j][i];
              avg[j] /= NAVG;
            }
          if (nlat == 0) { latency = -1; nlat = 1; }
          LINFO("%.1ffps/%llxms [G=%d D=%d S=%d R=%d A=%d T=%d F=%d]",
                1000.0 / ((float)(avg[5])), framepause,
                avg[0], avg[1], avg[2], avg[3], avg[4], avg[5],
                latency / nlat);
          latency = 0; nlat = 0;
          // try to resorb the number of dropped frames:
          if (dropped) dropped --;
        }
      t[4][ff] = tim.get();  // after total before pause

      // if necessary, sleep for a while until fixed amount of time elapsed:
      while(tim.get() < framepause)
        {
          struct timespec ts, ts2;
          ts.tv_sec = 0; ts.tv_nsec = 1000000;  // sleep 1 ms
          nanosleep(&ts, &ts2);
        }

      t[5][ff] = tim.get();  // total time

      // ready for next frame:
      frame++;
    }

  // got interrupted; let's cleanup and exit:
  // stop all our ModelComponents
  manager.stop();
  exit(0);
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
