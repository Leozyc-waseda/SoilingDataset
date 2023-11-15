/*!@file Parallel/pvisionTCP2-master.C Grab & process over beowulf w/ pvisionTCP2 */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Parallel/pvisionTCP2-master.C $
// $Id: pvisionTCP2-master.C 9412 2008-03-10 23:10:15Z farhan $
//

/*! This parallel vision processing master is for use with pvisionTCP2.
  See the pvisionTCP2go script in bin/ for how to launch the slaves.
*/

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"   // for inplacePaste()
#include "Image/DrawOps.H"
#include "Image/FilterOps.H"
#include "Image/Image.H"
#include "Image/Image.H"
#include "Image/ImageSet.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"   // for decX() etc.
#include "Parallel/pvisionTCP-defs.H"
#include "Transport/FrameIstream.H"
#include "Util/Assert.H"
#include "Util/Timer.H"

#include <signal.h>
#include <unistd.h>


//! Number of frames over which average framerate is computed
#define NAVG 20
//! Number of stats collected
#define NSTAT 4

//! prescale level by which we downsize images before sending them off
#define PRESCALE 2

//! shift attention at most once every SHIFTINT frames
#define SHIFTINT 5

//! injection ratio between new and older sm:
#define SMINJECT 0.6f

//! threshold salience (between 0 and 255) to warrant a shift of attention
#define SMTHRESH 200

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
  if (manager.parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

  // do post-command-line configs:
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
           "to be useful");
  int w = gb->getWidth(), h = gb->getHeight();

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // get prepared to grab, communicate, display, etc:
  XWindow xw(Dims(w * 2, h), -1, -1, "USC Saliency Cam");
  Image<PixRGB<byte> > disp(w * 2, h, ZEROS);
  int32 frame = 0;                // count the frames
  PixRGB<byte> pix(255, 255, 0);  // yellow color for fixations
  Point2D<int> win(w/2, h/2), winsm; // coordinates of attended location
  int foa_size = std::min(w, h) / 12; // radius of focus of attention
  TCPmessage rmsg;      // buffer to receive messages from nodes
  TCPmessage smsg;      // buffer to send messages to nodes
  Timer tim;            // for computation of framerate
  int t[NSTAT][NAVG], ff; // to compute average framerate and other stats
  int latency = 0, nlat = 0;  // to compute average processing latency
  int lastshift = 0;    // frame of last shift
  int sml = 4; Image<byte> sm;

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
      Image< PixRGB<byte> > ima = gb->readRGB();
      t[0][ff] = tim.getReset();  // grab time

      // send every other image to processing:
      if (frame & 1)
        {
          // prescale image:
          Image<PixRGB<byte> > ima2 =
            decY(lowPass9y(decX(lowPass9x(ima),1<<PRESCALE)),1<<PRESCALE);

          // we do the job of BEO_RETINA on the master to reduce latency:
          // compute luminance and send it off:
          Image<byte> lum = luminance(ima2);
          smsg.reset(frame, BEO_LUMINANCE);
          smsg.addImage(lum);
          beo->send(0, smsg);  // send off to luminance slave

          // compute RG and BY and send them off:
          Image<byte> r, g, b, y; getRGBY(ima2, r, g, b, y, (byte)25);
          smsg.reset(frame, BEO_REDGREEN);
          smsg.addImage(r); smsg.addImage(g);
          beo->send(1, smsg);  // send off to RG slave
          smsg.reset(frame, BEO_BLUEYELLOW);
          smsg.addImage(b); smsg.addImage(y);
          beo->send(2, smsg);  // send off to BY slave

          t[1][ff] = tim.getReset();  // send out time

          //LINFO("sent frame %d", frame);
        }
      else
        {
          // sleep a bit so that we give more CPU to other threads/processes:
          struct timespec ts, ts2;
          ts.tv_sec = 0; ts.tv_nsec = 5000000;  // sleep 5 ms
          nanosleep(&ts, &ts2);

          // duplicate real send time from previous frame:
          int ff2 = ff - 1; if (ff2 < 0) ff2 += NAVG;
          t[1][ff] = t[1][ff2]; tim.reset();
        }

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
                Image<byte> smap = rmsg.getElementByteIma();

                // trigger IOR at currently-attended location:
                if (sm.initialized() && rframe - lastshift <= SHIFTINT)
                  drawDisk(sm, winsm,
                           int(ceil(float(foa_size) / (1<<sml))),
                           byte(0));

                // inject newly received saliency map:
                if (sm.initialized())
                  sm = sm * (1.1f - SMINJECT) + smap * SMINJECT;
                else
                  sm = smap;

                if (rframe - lastshift >= SHIFTINT)
                  {
                    // find most salient location:
                    byte maxval; findMax(sm, winsm, maxval);

                    // rescale winner coordinates according to PRESCALE:
                    win.i = winsm.i << sml;
                    win.i += int(((1<<(sml-1)) * float(rand()))/RAND_MAX);
                    win.j = winsm.j << sml;
                    win.j += int(((1<<(sml-1)) * float(rand()))/RAND_MAX);

                    // just did a shift of attention:
                    lastshift = rframe;
                  }
              }
              break;
            default: // ##############################
              LERROR("Bogus action %d -- IGNORING.", raction);
              break;
            }
          // limit number of receives, so we don't hold CPU too long:
          recnb ++; if (recnb > 3) break;
        }
      t[2][ff] = tim.getReset(); // receive time

      // display image in window:
      inplacePaste(disp, ima, Point2D<int>(0, 0));
      drawPatch(disp, win, 3, pix);
      drawCircle(disp, win, foa_size, pix, 2);
      if (sm.initialized())
        inplacePaste(disp,
                     toRGB(quickInterpolate(sm, 1<<sml)), Point2D<int>(w, 0));
      xw.drawImage(disp);

      t[3][ff] = tim.getReset();  // display time
      if (t[3][ff] > 20)
        LINFO("*** Display took %dms for frame %d", t[3][ff], frame);


      // compute and show framerate and stats over the last NAVG frames:
      if (ff == 1 && frame > 1)
        {
          float avg[NSTAT], tot = 0.0f;
          for (int j = 0; j < NSTAT; j ++)
            {
              avg[j] = 0.0f;
              for (int i = 0; i < NAVG; i ++)
                { avg[j] += t[j][i]; tot += t[j][i]; }
              avg[j] /= NAVG;
            }
          tot /= NAVG;
          if (nlat == 0) { latency = -1; nlat = 1; }
          LINFO("%.1ffps [G=%.1f S=%.1f R=%.1f D=%.1f T=%.1f F=%.1f]",
                1000.0f / tot, avg[0], avg[1],
                avg[2], avg[3], tot, float(latency) / nlat);
          latency = 0; nlat = 0;
        }

      // ready for next frame:
      frame++;
    }

  // got interrupted; let's cleanup and exit:

  // stop all our ModelComponents
  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
