/*!@file Demo/test-tracking.C Test tracking  */

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
// Primary maintainer for this file: Lior Elazary <elazary@usc.edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/SeaBee/test-TunePID.C $
// $Id: test-TunePID.C 14376 2011-01-11 02:44:34Z pez $
//

#include "Image/OpenCVUtil.H"  // must be first to avoid conflicting defs of int64, uint64

#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWinManaged.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Component/ModelManager.H"
#include "Devices/DeviceOpts.H"
#include "GUI/XWindow.H"
#include "Image/DrawOps.H"
#include "Image/CutPaste.H"
#include "Image/Image.H"
#include "Image/Pixels.H"
#include "Image/MathOps.H"
#include "Neuro/EnvVisualCortex.H"
#include "Media/FrameSeries.H"
#include "Media/MediaOpts.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "Raster/Raster.H"
#include "Util/Timer.H"
#include "Util/log.H"
#include "Util/MathFunctions.H"
#include "Learn/Bayes.H"
#include "Learn/BackpropNetwork.H"
#include "Envision/env_image_ops.h"
#include "Neuro/BeoHeadBrain.H"
#include "Image/ShapeOps.H"
#include "Devices/BeeSTEM.H"
#include "rutz/shared_ptr.h"


#include <ctype.h>
#include <deque>
#include <iterator>
#include <stdlib.h> // for atoi(), malloc(), free()
#include <string>
#include <vector>
#include <map>

#define UP 98
#define DOWN 104
#define RIGHT 102
#define LEFT 100


#define THRUSTER_UP_LEFT 1
#define THRUSTER_UP_RIGHT 3
#define THRUSTER_UP_BACK 2
#define THRUSTER_FWD_RIGHT 4
#define THRUSTER_FWD_LEFT 0

//ModelManager *mgr;
XWinManaged *xwin;
Timer timer;
Image<PixRGB<byte> > disp;
byte SmaxVal = 0;
int smap_level = -1;

bool debug = 0;
bool init_points = true;



//! Our own little BeeSTEMListener
class BeeSTEM_PID_Listener : public BeeSTEMListener
{
        public:
                BeeSTEM_PID_Listener(nub::soft_ref<BeeSTEM> &b) :
                        itsBeeStem(b)
        {
                pidHeading.reset(new PID<float>(0.1, 0, 0, -100.0, 100.0));
                pidPitch.reset(new PID<float>(0.1, 0, 0, -100.0, 100.0));
                pidRoll.reset(new PID<float>(0.1, 0, 0, -100.0, 100.0));

                thrust_h_left = 0;
                thrust_h_right = 0;
                thrust_v_left = 0;
                thrust_v_right = 0;
                thrust_v_back = 0;

                target_heading = 0;
        }

                virtual ~BeeSTEM_PID_Listener()
                { }

                //These set and turn off the PID values.
                void setTargetHeading(int _h) {
                        target_heading = _h;
                        heading_enable = true;

                        //Now turn off all the motors so we can restart the PID
                        thrust_h_left = 0;
                        thrust_h_right = 0;
                        thrust_v_left = 0;
                        thrust_v_right = 0;
                        thrust_v_back = 0;
                }

                void setTargetPitch(int _h) {
                        target_pitch = _h;
                        pitch_enable = true;

                        //Now turn off all the motors so we can restart the PID
                        thrust_h_left = 0;
                        thrust_h_right = 0;
                        thrust_v_left = 0;
                        thrust_v_right = 0;
                        thrust_v_back = 0;
                }

                void disableHeadingTarget() {
                        heading_enable = false;
                }

                void disablePitchTarget() {
                        pitch_enable = false;
                }

                void setPrintOnly(bool b) {
                        print_only = b;
                }

                int getPos()
                {
                        return itsCurrentPos;

                }



                virtual void event(const BeeSTEMEventType t, const unsigned char dat1,
                                const unsigned char dat2)
                {
                        LDEBUG("Event: %d dat1 = %d, dat2 = %d", int(t), dat1, dat2);

                        switch(t) {

                                case COMPASS_HEADING_EVENT:
                                        {
                                                int heading = (unsigned int)dat1*2;
                                                itsCurrentPos = heading;
                                                if (pidHeading.is_valid())
                                                {
                                                        float rot = 100*pidHeading->update(target_heading, (float)heading);
                                                        thrust_h_left+=(int)rot;
                                                        thrust_h_right-=(int)rot;
                                                        //LINFO("%f %i %i %i", rot, heading, thrust_h_left, thrust_h_right);
                                                        if(!print_only) {
                                                                //itsBeeStem->setThrust(THRUSTER_FWD_RIGHT, thrust_h_right);
                                                                //itsBeeStem->setThrust(THRUSTER_FWD_LEFT, thrust_h_left);
                                                        }
                                                }

                                        }
                                        break;  //Careful here! The heading is sent /2 because of byte limits
                                case COMPASS_PITCH_EVENT:
                                        {
                                                if(pitch_enable)
                                                {
                                                        int pitch = (signed char)dat1;
                                                        if (pidPitch.is_valid())
                                                        {
                                                                float pitchCorr = 100*pidPitch->update(target_pitch, (float)pitch);
                                                                thrust_v_left += (int)pitchCorr;
                                                                thrust_v_right -= (int)pitchCorr;
                                                                itsBeeStem->setMotor(THRUSTER_UP_RIGHT, (int)thrust_v_right);
                                                                itsBeeStem->setMotor(THRUSTER_UP_LEFT, (int)thrust_v_left);
                                                        }
                                                }
                                        }
                                        break;
                        case COMPASS_ROLL_EVENT:
                        {
                                int roll = (signed char)dat1;
                                roll++;
                        }
                        break;
                        case ACCEL_X_EVENT:           break;
                        case ACCEL_Y_EVENT:           break;
                        case INT_PRESS_EVENT:        /* LINFO("INTERNAL PRESSURE: %d", (int)dat1); */break;
                        case EXT_PRESS_EVENT:
                                                                                                                                                                                                                                                                                                                                         //        LINFO("%i", (unsigned char)dat1);
                                                                                                                                                                                                                                                                                                                                         break;
                        case TEMP1_EVENT:             break;
                        case TEMP2_EVENT:             break;
                        case DIG_IN_EVENT:            break;
                        case ADC_IN_EVENT:            break;
                        case MOTOR_A_CURR_EVENT:      break;
                        case MOTOR_B_CURR_EVENT:      break;
                        case MOTOR_C_CURR_EVENT:      break;
                        case MOTOR_D_CURR_EVENT:      break;
                        case MOTOR_E_CURR_EVENT:      break;
                        case ECHO_REPLY_EVENT:        LINFO("BeeSTEM Echo Reply Recieved."); break;
                        case RESET_EVENT:             LERROR("BeeSTEM RESET occurred!"); break;
                        case SW_OVERFLOW_EVENT:       LERROR("BeeSTEM Software Overflow!"); break;
                        case FRAMING_ERR_EVENT:       LERROR("BeeSTEM Framing Error!"); break;
                        case OVR_ERR_EVENT:           LERROR("BeeSTEM Hardware Overflow!"); break;
                        case HMR3300_LOST_EVENT:      break;
                        case ACCEL_LOST_EVENT:        break;
                        case TEMP1_LOST_EVENT:        break;
                        case TEMP2_LOST_EVENT:        break;
                                                                                                                                                //case TEMP3_LOST_EVENT:        break;
                        case ESTOP_EVENT:             break;
                        case UNRECOGNIZED_EVENT:      break;
                        case BAD_OUT_CMD_SEQ_EVENT:   LERROR("BeeSTEM Reports a Bad Command Sequence!"); break;
                        case BAD_IN_CMD_SEQ_EVENT:    break;
                        case RESET_ACK_EVENT:         LINFO("BeeSTEM Acknowledges Reset Request"); break;
                        case  NO_EVENT:               break;
                        default:                      LERROR("Unknown event %d received!", int(t)); break;

                }



}
private:
nub::soft_ref<BeeSTEM> itsBeeStem;
int heading;
bool heading_enable, print_only;
bool pitch_enable;
int target_pitch;

rutz::shared_ptr< PID<float> > pidHeading;
rutz::shared_ptr< PID<float> > pidPitch;
rutz::shared_ptr< PID<float> > pidRoll;
rutz::shared_ptr< PID<float> > pidDepth;

signed int thrust_h_left, thrust_h_right, thrust_v_left, thrust_v_right, thrust_v_back;

int target_heading;
int itsCurrentPos;
};


void display(Image<PixRGB<byte> > &img, Image<PixRGB<byte> > &posValImg,
                int pressure, int heading, int tilt, int pan);

int main(int argc, const char **argv)
{
        // Instantiate a ModelManager:
        ModelManager *mgr = new ModelManager("SeaBee Interface");

        nub::ref<InputFrameSeries> ifs(new InputFrameSeries(*mgr));
        mgr->addSubComponent(ifs);

        nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(*mgr));
        mgr->addSubComponent(ofs);

        nub::soft_ref<BeeSTEM> b(new BeeSTEM(*mgr,"BeeSTEM", "BeeSTEM", "/dev/ttyS1"));
        mgr->addSubComponent(b);

        //Make the PID Listener
        rutz::shared_ptr<BeeSTEM_PID_Listener> PID_Listener(new BeeSTEM_PID_Listener(b));
        rutz::shared_ptr<BeeSTEMListener> lis2; lis2.dynCastFrom(PID_Listener); // cast down
        b->setListener(lis2);

        mgr->exportOptions(MC_RECURSE);

        mgr->setOptionValString(&OPT_InputFrameSource, "V4L2");
        mgr->setOptionValString(&OPT_FrameGrabberMode, "YUYV");
        mgr->setOptionValString(&OPT_FrameGrabberDims, "1024x576");
        mgr->setOptionValString(&OPT_FrameGrabberByteSwap, "no");
        mgr->setOptionValString(&OPT_FrameGrabberFPS, "30");

        // Parse command-line:
        if (mgr->parseCommandLine(argc, argv, "", 0, 0) == false) return(1);

        // do post-command-line configs:
        Dims imageDims(320,240);

        xwin = new XWinManaged(Dims(imageDims.w()*2,imageDims.h()*2+20),
                        -1, -1, "SeaBee interface");
        disp = Image<PixRGB<byte> >(imageDims.w(),imageDims.h()+20, ZEROS);

        // let's get all our ModelComponent instances started:
        mgr->start();

        //start streaming
        ifs->startStream();

        timer.reset();

        int target_heading = 100;

        b->setReporting(HMR3300, true);
        b->setReporting(INT_PRESS, true);

        int x = 0;
        Image<PixRGB<byte> > posValImg(256, 256, ZEROS);

        while(1) {

                const FrameState is = ifs->updateNext();
                if (is == FRAME_COMPLETE)
                        break;

                //grab the images
                GenericFrame input = ifs->readFrame();
                if (!input.initialized())
                        break;
                Image<PixRGB<byte> > frontImg = rescale(input.asRgb(), 320, 240);

                ofs->writeRGB(frontImg, "input", FrameInfo("Copy of input", SRC_POS));


                int key = xwin->getLastKeyPress();

                int val = PID_Listener->getPos();
                int y = (256/2) + ((target_heading - val));

                if (!x)
                {
                        posValImg.clear();
                        drawLine(posValImg, Point2D<int>(0, 256/2), Point2D<int>(256, 256/2), PixRGB<byte>(255,0,0));
                }
                LINFO("%i %i %i", val, (val - target_heading), y);
                posValImg.setVal(x,y,PixRGB<byte>(0,256,0));
                x = (x+1)%256;


                int speed = 100;
                switch(key)
                {
                        case 38: //a
                                b->setMotor(2,-1 * speed);
                                b->setMotor(1,-1 * speed);
                                b->setMotor(3,-1 * speed);
                                break;
                        case 52: //z
                                b->setMotor(2,speed );
                                b->setMotor(1,speed );
                                b->setMotor(3,speed );
                                break;
                        case LEFT:
                                b->setMotor(0, 1 * speed);
                                b->setMotor(4, -1 * speed);
                                break;
                        case RIGHT:
                                //target_heading++;
                                //PID_Listener->setTargetHeading(target_heading);
                                b->setMotor(0, -1 * speed);
                                b->setMotor(4, 1 * speed);
                                break;
                        case UP:
                                b->setMotor(0, -1 * speed);
                                b->setMotor(4, -1 * speed);
                                break;
                        case DOWN:
                                b->setMotor(0, speed);
                                b->setMotor(4, speed);
                                break;
                        case 65:  //space
                                b->setMotor(0,0);
                                usleep(1000);
                                b->setMotor(1,0);
                                usleep(1000);
                                b->setMotor(2,0);
                                usleep(1000);
                                b->setMotor(3,0);
                                usleep(1000);
                                b->setMotor(4,0);
                                usleep(1000);
                                break;
                        case 33: //p
                                PID_Listener->setTargetHeading((signed char)b->getCompassPitch());
                                break;
                        default:
                                if (key != -1)
                                        LINFO("Key = %i", key);
                                break;
                }
                display(frontImg, posValImg, (int)b->getIntPress(), (int)b->getCompassHeading()*2, (signed char)b->getCompassPitch(), (signed char)b->getCompassRoll());

        }


        // stop all our ModelComponents
        mgr->stop();

        // all done!
        return 0;
}

void display(Image<PixRGB<byte> > &img, Image<PixRGB<byte> > &posValImg,
                int pressure, int heading, int tilt, int pan)
{
        static int avgn = 0;
        static uint64 avgtime = 0;
        char msg[255];

        //Left Image
        xwin->drawImage(img, 0, 0);
        xwin->drawImage(posValImg, img.getWidth()+1, 0);

        //calculate fps
        avgn++;
        avgtime += timer.getReset();
        if (avgn == 20)
        {
                avgtime = 0;
                avgn = 0;
        }

        sprintf(msg, "pre=%i\t head=%i\t tilt=%i\t pan=%i\t ",
                        pressure, heading, tilt, pan);



        Image<PixRGB<byte> > infoImg(img.getWidth()*2, 20, NO_INIT);
        writeText(infoImg, Point2D<int>(0,0), msg,
                        PixRGB<byte>(255), PixRGB<byte>(127));
        xwin->drawImage(infoImg, 0,img.getHeight()*2);

}

