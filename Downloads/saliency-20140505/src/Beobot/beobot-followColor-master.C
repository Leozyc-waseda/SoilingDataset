/*!@file Beobot/beobot-followColor-master.C color segment following - master
  adapted from VFAT/test-segmentImageMC.C                               */

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
// Primary maintainer for this file:  Christian Siagian <siagian@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-followColor-master.C $
// $Id: beobot-followColor-master.C 7353 2006-10-30 15:27:31Z rjpeters $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Transport/FrameIstream.H"
#include "GUI/XWinManaged.H"
#include "GUI/XWindow.H"

#include "Beobot/BeobotControl.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "VFAT/segmentImageTrackMC.H"
#include <cstdio>
#include <cstdlib>
#include <signal.h>

#include "Beobot/BeobotConfig.H"

// number of frames over which framerate info is averaged:
#define NAVG 20

static bool goforever = true;

// ######################################################################
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
//! The main routine. Grab frames, process, send commands to slave node.
int main(const int argc, const char **argv)
{
  // instantiate a model manager
  ModelManager manager("Follow Color Segments - Master");

//   nub::soft_ref<Beowulf>
//     beo(new Beowulf(manager, "Beowulf Master", "BeowulfMaster", true));
//   manager.addSubComponent(beo);

  BeobotConfig bbc;
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  nub::soft_ref<FrameGrabberConfigurator>
    gbc(new FrameGrabberConfigurator(manager));
  manager.addSubComponent(gbc);

//   manager.exportOptions(MC_RECURSE);

//   // frame grabber setup
//   // NOTE: don't have to put the option --fg-type=1394
//   manager.setOptionValString(&OPT_FrameGrabberType, "1394");
//   manager.setOptionValString(&OPT_FrameGrabberDims, "160x120");
//   manager.setOptionValString(&OPT_FrameGrabberMode, "YUV444");
//   manager.setOptionValString(&OPT_FrameGrabberNbuf, "20");

  // parse command-line
  if( manager.parseCommandLine( argc, argv, "", 0, 0 ) == false ) return(1);

  // let's configure our serial device:
  b->setModelParamVal("BeoChipDeviceName", std::string("/dev/ttyS0"));
  // NOTE: may want to add a BeoChip listener

  // configure the camera
  nub::soft_ref<FrameIstream> gb = gbc->getFrameGrabber();
  if (gb.isInvalid())
    LFATAL("You need to select a frame grabber type via the "
           "--fg-type=XX command-line option for this program "
          "to be useful");
  int width = gb->getWidth(), height = gb->getHeight();

  // let's get all our ModelComponent instances started:
  manager.start();

  // get the frame grabber to start streaming:
  gb->startStream();

  // display window
  XWindow wini(Dims(width, height), 0, 0, "test-input window");
  XWindow wino(Dims(width/4, height/4), 0, 0, "test-output window 2");
  XWindow winAux(Dims(500, 450), 0, 0, "Channel levels");
  Image< PixRGB<byte> > ima; Image< PixRGB<float> > fima;
  Image< PixRGB<byte> > display;
  Image<PixH2SV2<float> > H2SVimage;

  TCPmessage rmsg;      // buffer to receive messages from nodes
  TCPmessage smsg;      // buffer to send messages to nodes

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // reset the beochip:
  LINFO("Resetting BeoChip...");
  b->resetChip(); sleep(1);

  // calibrate the servos
  b->calibrateServo(bbc.steerServoNum, bbc.steerNeutralVal,
                    bbc.steerMinVal, bbc.steerMaxVal);
  b->calibrateServo(bbc.speedServoNum, bbc.speedNeutralVal,
                    bbc.speedMinVal, bbc.speedMaxVal);
  b->calibrateServo(bbc.gearServoNum, bbc.gearNeutralVal,
                    bbc.gearMinVal, bbc.gearMaxVal);

  // keep the gear at the lowest speed/highest Ntorque
  b->setServoRaw(bbc.gearServoNum, bbc.gearMinVal);

  // timer initialization
  Timer tim; Timer camPause;       // to pause the move command
  camPause.reset();
  uint64 t[NAVG]; int frame = 0;

  // ######################################################################
  //! extracted color signature

  // H1 - H2 - S - V
  std::vector<float> color(4,0.0F);
  //BLUE
  //  color[0] = 0.350962; color[1] = 0.645527;
  //color[2] = 0.313523; color[3] = 0.720654;

  // YELLOW
  //color[0] = 0.8; color[1] = 0.44;
  //color[2] = 0.65; color[3] = 0.82;

  // RED
  color[0] = 0.75; color[1] = 0.87;
  color[2] = 0.48; color[3] = 0.70;

  //! +/- tollerance value on mean for track
  std::vector<float> std(4,0.0F);
  std[0] = 0.339556; std[1] = 0.368726;
  std[2] = 0.609608; std[3] = 0.34012;

  //! normalizer over color values (highest value possible)
  std::vector<float> norm(4,0.0F);
  norm[0] = 1.0F; norm[1] = 1.0F;
  norm[2] = 1.0F; norm[3] = 1.0F;

  //! how many standard deviations out to adapt, higher means less bias
  std::vector<float> adapt(4,0.0F);
  adapt[0] = 3.5F; adapt[1] = 3.5F;
  adapt[2] = 3.5F; adapt[3] = 3.5F;

  //! highest value for color adaptation possible (hard boundary)
  std::vector<float> upperBound(4,0.0F);
  upperBound[0] = color[0] + 0.45F; upperBound[1] = color[1] + 0.45F;
  upperBound[2] = color[2] + 0.55F; upperBound[3] = color[3] + 0.55F;

  //! lowest value for color adaptation possible (hard boundary)
  std::vector<float> lowerBound(4,0.0F);
  lowerBound[0] = color[0] - 0.45F; lowerBound[1] = color[1] - 0.45F;
  lowerBound[2] = color[2] - 0.55F; lowerBound[3] = color[3] - 0.55F;

  int wi = width/4;  int hi = height/4;
  segmentImageTrackMC<float,unsigned int, 4> segmenter(wi*hi);
  segmenter.SITsetTrackColor(&color,&std,&norm,&adapt,&upperBound,&lowerBound);

  // Limit area of consideration to within the image
  segmenter.SITsetFrame(&wi,&hi);

  // Set display colors for output of tracking. Strictly asthetic
  segmenter.SITsetCircleColor(0,255,0);
  segmenter.SITsetBoxColor(255,255,0,0,255,255);
  segmenter.SITsetUseSmoothing(true,10);
  // ######################################################################

  // get ready for main loop:
  //  int32 rframe = 0, raction = 0;
  while(goforever)
  {
    tim.reset();
    ima = gb->readRGB();
    //uint64 t0 = tim.get();  // to measure display time

    // segment image on each input frame
    Image<PixRGB<byte> > Aux; Aux.resize(100,450,true);
    H2SVimage = ima;    // NOTE: this line takes 50ms.
    display = ima;
    segmenter.SITtrackImageAny(H2SVimage,&display,&Aux,true);

    // Retrieve and Draw all our output images
    Image<byte> temp = segmenter.SITreturnCandidateImage();
    wini.drawImage(display);
    wino.drawImage(temp);
    winAux.drawImage(Aux);

    // if we find the tracked object
    float st,sp,gr;
    if(true || !segmenter.SITreturnLOT())
      {
        int x, y, m; unsigned int minX, maxX, minY, maxY;
        segmenter.SITgetBlobPosition(x,y);
        segmenter.SITgetBlobWeight(m);
        segmenter.SITgetMinMaxBoundry(&minX, &maxX, &minY, &maxY);
        LINFO("x = %d y = %d m = %d", x, y, m);
        LDEBUG("[%3d %3d %3d %3d] -> %d",
               minX, maxX, minY, maxY,
               (maxX - minX) * (maxY - minY));

        int nBlobs = segmenter.SITnumberBlobs();
        for(int i = 0; i < nBlobs; i++)
          {
            int bx = segmenter.SITgetBlobPosX(i);
            int by = segmenter.SITgetBlobPosY(i);
            int bm = segmenter.SITgetBlobMass(i);
            LDEBUG("bx = %d by = %d bm = %d", bx, by, bm);
          }

        // steer is controlled by blob x-coor
        st = 0.0f;
        if(x < 120)      st = -0.8f;
        else if(x > 200) st =  0.8f;
        else             st = float((x - 120.0)/80.0*1.6 - 0.8);

        // speed is controlled by size of blob
        if(m <= 200)
          { sp = 0.3f; st = 0.0f; }
        if(m > 200 && m < 2000)
          sp = (m - 200.0) / 1800.0 * 0.4;
        else if(m >= 2000 && m < 6000)
          sp = (8000.0 - m) / 6000.0f * 0.4;
        else if(m >= 6000 && m < 10000)
          { sp = (6000.0 - m)/4000.0 * 0.3; st = 0.0; }
        else  { sp = 0.3f; st = 0.0f; }

        // stay at the lowest gear: hi torque/low speed
        gr = 0.0;
      }
    else
      {
        LINFO("lost of track - will look for it");

        st = 0.0;
        sp = 0.4;
        gr = 0.0;
      }
    LINFO("st = %f sp = %f gr = %f", st, sp, gr);

    // make sure command is not too taxing on the motors
    // can only send suggestion to stop or move faster
    // do we need to delay sending commands?

    // send action command to Board B
//     smsg.reset(rframe, raction);
//     smsg.addFloat(st); smsg.addFloat(sp); smsg.addFloat(gr);
//     beo->send(0, smsg);
      b->setServo(bbc.steerServoNum, st);
      b->setServo(bbc.speedServoNum, sp);

    // compute and show framerate over the last NAVG frames:
    t[frame % NAVG] = tim.get();
    //t0 = t[frame % NAVG] - t0; if (t0 > 28) LINFO("Display took %llums", t0);

    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000.0 / (float)avg * NAVG;
        printf("Framerate: %.1f fps\n", avg2);
      }
    frame ++;
  }

  manager.stop();
  return 0;
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
