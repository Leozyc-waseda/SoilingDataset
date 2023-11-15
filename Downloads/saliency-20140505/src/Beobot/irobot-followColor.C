/*!@file Beobot/irobot-followColor.C Test color segment following */

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
// Primary maintainer for this file:  T. Moulard <thomas.moulard@gmail.com>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/irobot-followColor.C $
// $Id: irobot-followColor.C 13993 2010-09-20 04:54:23Z itti $
//



// model manager
#include "Component/ModelManager.H"
#include "Util/log.H"
#include "rutz/shared_ptr.h"

// for images and display
#include "Raster/Raster.H"
#include "Image/Image.H"
#include "Image/PixelsTypes.H"
#include "Transport/FrameIstream.H"
#include "GUI/XWinManaged.H"
#include "GUI/XWindow.H"

// Frame grabber
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "Media/FrameSeries.H"

// for color segmentation
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "VFAT/segmentImageTrackMC.H"
#include <cstdio>
#include <cstdlib>
#include <signal.h>

// for image manipulation
#include "Image/CutPaste.H"     // for inplacePaste()

// number of frames over which framerate info is averaged:
#define NAVG 20

#if HAVE_LIBSERIAL && HAVE_LIBIROBOT_CREATE

static bool goforever = true;

// for Robot controller
#include <SerialStream.h>
#include <irobot-create.hh>

// ######################################################################
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
//! Receive signals from master node and performs requested actions
int main(const int argc, const char **argv)
{
  using namespace iRobot;
  using namespace LibSerial;

  MYLOGVERB = LOG_INFO;

  // instantiate a model manager
  ModelManager manager( "Following Color Segments " );

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.setOptionValString(&OPT_FrameGrabberMode, "RGB24");
  manager.setOptionValString(&OPT_FrameGrabberDims, "320x240");
  manager.setOptionValString(&OPT_FrameGrabberFPS, "30");

  manager.exportOptions(MC_RECURSE);

  // parse command-line
  if( manager.parseCommandLine( argc, argv, "", 0, 0 ) == false ) return(1);

  manager.start();

  // get a frame grabber
  Dims imageDims = ifs->peekDims();
  uint width  = imageDims.w();
  uint height = imageDims.h();


  // initialize the motor controller
  SerialStream stream(std::string("/dev/rfcomm1"), std::ios::in | std::ios::out);

  // display window
  XWindow wini(Dims(width, height), 0, 0, "test-input window");
  XWindow wino(Dims(width/4, height/4), 650, 0, "test-output window 2");
  XWindow winAux(Dims(500, 450), 1000, 0, "Channel levels");
  Image< PixRGB<byte> > ima; Image< PixRGB<float> > fima;
  Image< PixRGB<byte> > display;
  Image<PixH2SV2<float> > H2SVimage;

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

  //! +/- tolerance value on mean for track
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

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate); signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  // instantiate a robot controller
  Create robot(stream);

  // Swith to full mode.
  robot.sendFullCommand();

  // Let's stream some sensors.
  Create::sensorPackets_t sensors;
  sensors.push_back(Create::SENSOR_BUMPS_WHEELS_DROPS);
  sensors.push_back(Create::SENSOR_WALL);
  sensors.push_back(Create::SENSOR_BUTTONS);
  robot.sendStreamCommand(sensors);

  // Let's turn!
  int speed = 200;
  //int ledColor = Create::LED_COLOR_GREEN;
  robot.sendDriveCommand(speed, Create::DRIVE_INPLACE_CLOCKWISE);
  robot.sendLedCommand(Create::LED_PLAY, 0, 0);

  // timer initialization
  Timer tim; Timer camPause;       // to pause the move command
  camPause.reset();
  //uint64 t[NAVG];
  uint fNum = 0;

  // get ready for main loop:
  while (!robot.playButton () && goforever)
    {
      // performace monitor
      tim.reset();

      // process robot input
      if (robot.bumpLeft () || robot.bumpRight ())
      {

        std::cout << "Bump !" << std::endl;

        // take a picture


      }


      //   if (robot.wall ())        std::cout << "Wall !" << std::endl;

     if (robot.advanceButton ())
        {

        }

      // get, convert, and display image
      ima = ifs->readRGB();
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

      // write in all the info
      Image<PixRGB<byte> > dispAux(500,450, ZEROS);
      inplacePaste(dispAux, Aux, Point2D<int>(0, 0));

      // if we find the tracked object
      float st,sp;
      if(true || !segmenter.SITreturnLOT())
        {
          int x, y, m; unsigned int minX, maxX, minY, maxY;
          segmenter.SITgetBlobPosition(x,y);
          segmenter.SITgetBlobWeight(m);
          segmenter.SITgetMinMaxBoundry(&minX, &maxX, &minY, &maxY);
          //LINFO("x = %d y = %d m = %d", x, y, m);
          std::string ntext(sformat("x = %5d y = %5d m = %6d", x, y, m));
          writeText(dispAux, Point2D<int>(100, 0), ntext.c_str());


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

//           speed = -1 * speed;
//           ledColor += 10;
//           if (ledColor > 255)
//             ledColor = 0;

//           robot.sendDriveCommand (speed, Create::DRIVE_INPLACE_CLOCKWISE);
//           if (speed < 0)
//             robot.sendLedCommand (Create::LED_PLAY,
//                                   ledColor,
//                                   Create::LED_INTENSITY_FULL);
//           else
//             robot.sendLedCommand (Create::LED_ADVANCE,
//                                   ledColor,
//                                   Create::LED_INTENSITY_FULL);


        }
      else
        {
          std::string ntext("lost of track - will look for it");
          writeText(dispAux, Point2D<int>(100, 80), ntext.c_str());
          //LINFO("lost of track - will look for it");

          st = 0.0;
          sp = 0.4;
        }


      // move robot
      std::string ntext3(sformat("REAL sp = %f st = %f",
                                 sp * 200.0, st * 2000.0));
      writeText(dispAux, Point2D<int>(100, 120), ntext3.c_str());
      try
        {
          short radius = 0;
          if (st < 0)
            radius = 2000. + (st * 2000.0);
          else if (st > 0)
            radius = -2000. + (st * 2000.0);
          robot.sendDriveCommand (sp * 200.0, radius);
        }
      catch (...) {}

      std::string ntext2(sformat("st = %f sp = %f", st, sp));
      writeText(dispAux, Point2D<int>(100, 40), ntext2.c_str());
      //LINFO("st = %f sp = %f gr = %f", st, sp, gr);

      winAux.drawImage(dispAux);

      // You can add more commands here.
      usleep(20 * 1000);

  LINFO("I cannot work without LibSerial or libirobot-create");

      fNum++;
    }

  robot.sendDriveCommand (0, Create::DRIVE_STRAIGHT);

}

#else

int main(const int argc, const char **argv)
{
  //LINFO("I cannot work without LibSerial or libirobot-create");
  return 1;
}

#endif

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
