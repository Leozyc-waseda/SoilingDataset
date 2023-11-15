/*!@file Beobot/beobot-driveStraight-master.C color drive straight - master
  adapted from RCBot/driveStraight.C                               */
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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Beobot/beobot-driveStraight-master.C $
// $Id: beobot-driveStraight-master.C 9412 2008-03-10 23:10:15Z farhan $
//

#include "Beowulf/Beowulf.H"
#include "Component/ModelManager.H"
#include "Component/OptionManager.H"
#include "Controllers/PID.H"
#include "Devices/FrameGrabberConfigurator.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWindow.H"
#include "Image/ColorOps.H"
#include "Image/CutPaste.H"
#include "Image/DrawOps.H"
#include "Image/Image.H"
#include "Image/MathOps.H"
#include "Image/Pixels.H"
#include "Image/ShapeOps.H"
#include "RCBot/Motion/MotionEnergy.H"
#include "Raster/Raster.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameIstream.H"
#include "Util/Timer.H"
#include "Util/Types.H"
#include "Util/log.H"
#include <math.h>
#include <signal.h>

#include "Devices/BeoChip.H"
#include "Beobot/BeobotConfig.H"
#include "Controllers/PID.H"
#include "Beobot/BeobotBeoChipListener.H"

// number of frames over which framerate info is averaged:
#define NAVG 20

static bool goforever = true;
BeobotConfig bbc;

// ######################################################################
void resetBeoChip(nub::soft_ref<BeoChip> b);

// ######################################################################
void terminate(int s)
{ LERROR("*** INTERRUPT ***"); goforever = false; exit(1); }

// ######################################################################
int main(const int argc, const char **argv)
{
  MYLOGVERB = LOG_INFO;

  // instantiate a model manager:
  ModelManager manager("Beobot - Drive Straight");

  // Instantiate our various ModelComponents:
  nub::soft_ref<BeoChip> b(new BeoChip(manager));
  manager.addSubComponent(b);

  nub::soft_ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);

  manager.exportOptions(MC_RECURSE);

  // Parse command-line:
  if (manager.parseCommandLine(argc, argv, "steer P I D speed P I D", 6, 6) == false)
    return(1);

  // do post-command-line configs:

  // configure the BeoChip
  b->setModelParamVal("BeoChipDeviceName", std::string("/dev/ttyS0"));

  // let's register our listener:
  rutz::shared_ptr<BeobotBeoChipListener> lis(new BeobotBeoChipListener(b));
  rutz::shared_ptr<BeoChipListener> lis2; lis2.dynCastFrom(lis); // cast down
  b->setListener(lis2);

  int w = ifs->getWidth(), h = ifs->getHeight();
  XWindow winoH(Dims(w, 4*h), 2*w+20, 0, "test-output H");
  XWindow winoV(Dims(w, 4*h), 3*w+30, 0, "test-output V");
  XWindow wini (Dims(w, 4*h), w+10  , 0, "test-input windowL");
  XWindow wini2(Dims(w, 4*h), 0     , 0, "test-input windowC");

  // let's get all our ModelComponent instances started:
  manager.start();

  // reset BeoChip
  resetBeoChip(b);

  // create the motion pyramid
  MotionEnergyPyrBuilder<byte> motionPyr(Gaussian5);

  // catch signals and redirect them to terminate for clean exit:
  signal(SIGHUP, terminate);  signal(SIGINT, terminate);
  signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
  signal(SIGALRM, terminate);

  PID<float> steer_pid(4.0, 0.0, 0.0, -1.0, 1.0);
  PID<float> speed_pid(1.6, 0.3, 0.3, -0.6, 0.6);
  steer_pid.setPIDPgain(manager.getExtraArgAs<float>(0));
  steer_pid.setPIDIgain(manager.getExtraArgAs<float>(1));
  steer_pid.setPIDDgain(manager.getExtraArgAs<float>(2));
  speed_pid.setPIDPgain(manager.getExtraArgAs<float>(3));
  speed_pid.setPIDIgain(manager.getExtraArgAs<float>(4));
  speed_pid.setPIDDgain(manager.getExtraArgAs<float>(5));

  // timer initialization
  Timer tim(1000000); uint64 t[NAVG]; int frame = 0;
  // ########## MAIN LOOP: grab, process, display:
  while(goforever)
  {
    tim.reset();

    // grab an image:
    ifs->updateNext();
    Image< PixRGB<byte> > ima = ifs->readRGB();
    if(!ima.initialized()) {goforever = false; break;}
    wini2.drawImage(ima,0,0);
//     Image< PixRGB<byte> > tmp2 = decXY(ima);
//     wini2.drawImage(tmp2,0,h);
//     tmp2 = decXY(tmp2);
//     wini2.drawImage(tmp2,0,h+h/2);
//     tmp2 = decXY(tmp2);
//     wini2.drawImage(tmp2,0,h+h/2+h/4);

    //Image<byte> lum(320,240,ZEROS);
    //drawPatch(lum, Point2D<int>(160, int(120-frame*.5)),10, byte(255));
    //drawPatch(lum, Point2D<int>(int(120-frame*.5), 120),10, byte(255));
    Image<byte> lum = luminance(ima);
    wini.drawImage(lum,0,0);
//     Image<byte> tmp = decXY(lum);
//     wini.drawImage(tmp,0,h);
//     tmp = decXY(tmp);
//     wini.drawImage(tmp,0,h+h/2);
//     tmp = decXY(tmp);
//     wini.drawImage(tmp,0,h+h/2+h/4);

    // build the motion pyramid
    // FIX: the second arg is depth of the pyramid
    //      figure out how to combine each depth info later
    motionPyr.updateMotion(lum, 1); // start: 1
    ImageSet<float> hpyr = motionPyr.buildHorizontalMotion();
    ImageSet<float> vpyr = motionPyr.buildVerticalMotion();

    // draw the images
    int sc = 1; float mn,mx, mn2, mx2;
    for(uint i = 0; i < hpyr.size(); i++)
      {
        Image<float> tmp = hpyr[i]; getMinMax(tmp,mn,mx);
        for(uint j = 0; j < i; j++) tmp = intXY(tmp, true);
        winoH.drawImage(tmp, 0, h*i);

        tmp = vpyr[i]; getMinMax(tmp,mn2,mx2);
        for(uint j = 0; j < i; j++) tmp = intXY(tmp,true);
        winoV.drawImage(tmp, 0, h*i);
        sc *=2;

//         LINFO("%4d| m(h): [%7.4f %7.4f %7.4f] m(v): [%7.4f %7.4f %7.4f]", frame,
//               mn,  mean(hpyr[i]), mx,
//               mn2, mean(vpyr[i]), mx2);
      }


    // ######################################################################
    // Obstacle avoidance
    // ######################################################################
//     Image<float> motion = vpyr[0];
//     float motionLeft = 0, motionRight = 0;
//     Image<float>::iterator motionPtr = motion.beginw();
//     Image<float>::const_iterator motionPtrStop = motion.end();

//     // find the motion energy on the left and right side
//     int inx = 0;
//     while (motionPtr != motionPtrStop) {
//       int y = inx / motion.getWidth();
//       int x = inx - (y*motion.getWidth());

//       if (y > 1){
//         if (x < (motion.getWidth()/2))
//           motionLeft  += fabs(*motionPtr);
//         else
//           motionRight += fabs(*motionPtr);
//       }
//       motionPtr++;
//       inx++;
//     }

//     double val = motionRight + motionLeft;
//     LINFO("Right: %0.4f Left: %0.4f Total: %0.4f",
//           motionRight, motionLeft, val);

//     if (val > 20)
//     {
//       if (motionLeft > motionRight)
//       {
//         drawLine(lum, Point2D<int>(64,64), Point2D<int>(64+30,64-30), (byte)0, 2);
//         b->setServo(bbc.steerServoNum, -1.0); //sc8000->move(1, -1);
//       }
//       else
//       {
//         drawLine(lum, Point2D<int>(64,64), Point2D<int>(64-30,64-30), (byte)0, 2);
//         b->setServo(bbc.steerServoNum, 1.0);  //sc8000->move(1, 1);
//       }
//     }
//     else
//       b->setServo(bbc.steerServoNum, 0.0); //sc8000->move(1, 0);

//     if (val > 4000)
//     {
//       LINFO("\n\nSTOP STOP STOP STOP \n");
//       b->setServo(bbc.speedServoNum, 0.0); //sc8000->move(3, 0);
//       sleep(2);
//     }
//     else
//       b->setServo(bbc.speedServoNum, -0.270); //sc8000->move(3, -0.270);

//     inplaceNormalize(motion, 0.0F, 255.0F);
//     winoV.drawImage((Image<byte>)motion, lum.getWidth()+2, 0);

    // ######################################################################
    // drive straight
    // ######################################################################

    double speed = fabs(mean(vpyr[0]));
    double dir   = -mean(hpyr[0]);
    if(dir > 0.0) LINFO("GO RIGHT"); else LINFO("GO LEFT");
    //    Raster::waitForKey();

//     drawLine(lum, Point2D<int>(w/2, h/2),
//              Point2D<int>( (int)(w/2 + 75*cos(dir)),
//                       (int)(h/2 - 75*sin(dir)) ),
//              (byte)0, 3);
//     window1.drawImage(rescale(lum, 256, 256));

    double st = steer_pid.update(0.0  , dir);
    double sp = speed_pid.update(0.175, speed);
    if(st > 1.0) st = 1.0; else if (st < -1.0) st = -1.0;
    if(sp > 0.6) sp = 0.6; else if (sp <  0.0) sp =  0.0;
    LINFO("%4d STEER: %11.7f -> %11.7f  SPEED: %11.7f -> %11.7f", frame,
          dir , st, speed, sp);
    sp = 0.4f;

    // execute action command
    b->setServo(bbc.steerServoNum, st);
    b->setServo(bbc.speedServoNum, sp);

    // ######################################################################
    // ######################################################################

    // compute and show framerate over the last NAVG frames:
    t[frame % NAVG] = tim.get();
    if (frame % NAVG == 0 && frame > 0)
      {
        uint64 avg = 0; for (int i = 0; i < NAVG; i ++) avg += t[i];
        float avg2 = 1000000.0 / (float)avg * NAVG;
        LINFO("Framerate: %5.2f fps", avg2);
      }
    frame++;
  }

  // got interrupted; let's cleanup and exit:
  manager.stop();
  return 0;
}

// ######################################################################
void resetBeoChip(nub::soft_ref<BeoChip> b)
{
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

  // keep the gear at the lowest speed/highest torque
  b->setServoRaw(bbc.gearServoNum, bbc.gearMinVal);

  // turn on the keyboard
  b->debounceKeyboard(true);
  b->captureKeyboard(true);

  // calibrate the PWMs:
  b->calibratePulse(0,
                    bbc.pwm0NeutralVal,
                    bbc.pwm0MinVal,
                    bbc.pwm0MaxVal);
  b->calibratePulse(1,
                    bbc.pwm1NeutralVal,
                    bbc.pwm1MinVal,
                    bbc.pwm1MaxVal);
  b->capturePulse(0, true);
  b->capturePulse(1, true);

  // let's play with the LCD:
  b->lcdClear();   // 01234567890123456789
  b->lcdPrintf(0, 0, "collectFrames: 00000");
  b->lcdPrintf(0, 1, "STEER=XXX  SPEED=XXX");
  b->lcdPrintf(0, 2, "PWM0=0000  0000-0000");
  b->lcdPrintf(0, 3, "PWM1=0000  0000-0000");
}

// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
