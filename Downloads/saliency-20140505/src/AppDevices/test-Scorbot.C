/*!@file AppDevices/test-Scorbot.C Test the scorbot robot arm controller */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/AppDevices/test-Scorbot.C $
// $Id: test-Scorbot.C 13961 2010-09-17 17:18:58Z lior $
//

#include "Component/ModelManager.H"
#include "Devices/Scorbot.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/PrefsWindow.H"
#include "Image/DrawOps.H"
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#define KEY_UP 98 
#define KEY_DOWN 104
#define KEY_LEFT 100
#define KEY_RIGHT 102


void home(nub::ref<OutputFrameSeries> &ofs);
void calMotor(Scorbot::MOTOR m,nub::ref<OutputFrameSeries> &ofs, int speed = 75);
nub::soft_ref<Scorbot> scorbot;
int motorStep = 0;
int motorCmd[10][6];

std::vector<Scorbot::ArmPos> armPositions;

void terminate(int s)
{
	LINFO("*** INTERRUPT ***");
	scorbot->stopAllMotors();
	scorbot->motorsOff();
	sleep(1);
	exit(0);
}

int getKey(nub::ref<OutputFrameSeries> &ofs)
{
	const nub::soft_ref<ImageDisplayStream> ids =
		ofs->findFrameDestType<ImageDisplayStream>();

	const rutz::shared_ptr<XWinManaged> uiwin =
		ids.is_valid()
		? ids->getWindow("Output")
		: rutz::shared_ptr<XWinManaged>();
	return uiwin->getLastKeyPress();
}

Image<PixRGB<byte> > getDisplay(bool motorsOn)
{
	char msg[255];
	Image<PixRGB<byte> > disp(512,256,ZEROS);
	disp.clear(PixRGB<byte>(255, 255, 255));
	writeText(disp, Point2D<int>(0,0), "Scorbot Control");
	sprintf(msg, "1-6 & q-y move joints");
	writeText(disp, Point2D<int>(0,20), msg);

	sprintf(msg, "p to show encoders, n to show micro switch, m to activate motors");
	writeText(disp, Point2D<int>(0,40), msg);

	sprintf(msg, "Motors state %i", motorsOn );
	writeText(disp, Point2D<int>(0,60), msg);

  return disp;
}



void moveToPositions()
{
//B:-3992 S:-1006 E:0 WR:121 WP:1595 W1:858 W2:-737 G:-2 E1:64 E2:0
//B:102 S:-1006 E:780 WR:-29 WP:1437 W1:704 W2:-733 G:-2 E1:76752 E2:0
//B:4526 S:-1006 E:780 WR:-17 WP:1765 W1:874 W2:-891 G:-2 E1:157468 E2:0
//B:4526 S:-3115 E:2845 WR:168 WP:1688 W1:928 W2:-760 G:-2 E1:157460 E2:0
//B:250 S:-3115 E:2846 WR:143 WP:1339 W1:741 W2:-598 G:-2 E1:79668 E2:0
//B:-4198 S:-3115 E:2846 WR:-33 WP:1747 W1:857 W2:-890 G:-2 E1:8 E2:0
//B:-4034 S:-4756 E:4318 WR:67 WP:1861 W1:964 W2:-897 G:-2 E1:16 E2:0
//B:-78 S:-4757 E:4320 WR:180 WP:1340 W1:760 W2:-580 G:-2 E1:75764 E2:0
//B:4432 S:-4758 E:4319 WR:257 WP:1847 W1:1052 W2:-795 G:-2 E1:157232 E2:0
//B:4432 S:-4822 E:7986 WR:-198 WP:2902 W1:1352 W2:-1550 G:-2 E1:157236 E2:0
//B:122 S:-4822 E:7986 WR:90 WP:2604 W1:1347 W2:-1257 G:-2 E1:78680 E2:0
//B:-4002 S:-4822 E:7986 WR:185 WP:3049 W1:1617 W2:-1432 G:-2 E1:28 E2:0


}


int main(int argc, const char **argv)
{
	// Instantiate a ModelManager:
	ModelManager manager("Test Model for Scorbot robot arm controller");

	nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
	manager.addSubComponent(ofs);

	scorbot = nub::soft_ref<Scorbot>(new Scorbot(manager));
	manager.addSubComponent(scorbot);


	// catch signals and redirect them to terminate for clean exit:
	signal(SIGHUP, terminate); signal(SIGINT, terminate);
	signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
	signal(SIGALRM, terminate);

	// Parse command-line:
	if (manager.parseCommandLine(argc, argv, "<Serial Data>", 0, 80) == false) return(1);


	// let's get all our ModelComponent instances started:
	manager.start();

  //Just send raw serial cmd to controller
	if (manager.numExtraArgs() > 0)
	{
    scorbot->shutdownMotorsOff(false); //Dont turn the motors off on shutdown

    int numChar = manager.numExtraArgs();
	  unsigned char buff[1024];
		for(int i=0; i<numChar; i++)
			buff[i] = manager.getExtraArgAs<int>(i);

		printf("Sending %i bytes: ", numChar);
		for(int i=0; i<numChar; i++)
			printf("%i ", buff[i]);
		printf("\n");


		int ret = scorbot->write(buff, numChar);
		usleep(50000);

		ret = scorbot->read(buff, 1024);
		printf("Got %i: ", ret);
		for(int i=0; i<ret; i++)
			printf("%3d(%3x) ", buff[i], buff[i]);
		printf("\n");

		manager.stop(); //Dont turn the motors off
		return 0;
	}

		
		

	//  //Example FK and IK
	//  Point3D<float> efPos = scorbot->getEFPos(-664, -12765);
	//  LINFO("EF Pos(2405, -3115) %f %f %f", efPos.x, efPos.y, efPos.z);
	//  Scorbot::ArmPos armPos = scorbot->getArmPos(efPos);
	//  LINFO("ArmPos: sholder %i elbow %i", armPos.sholder, armPos.elbow);
	//  return 0;


	//Pref win
	PrefsWindow pWin("Scorbot Control", SimpleFont::FIXED(8));
	pWin.setValueNumChars(16);
	pWin.addPrefsForComponent(scorbot.get(), false);



	int PGain = 0;
	int IGain = 0;
	int DGain = 0;

	int speed = 50;

	bool showMS = false;
	bool rawWrist = false;

	Scorbot::MOTOR pidMot = Scorbot::EX1;

	Scorbot::ArmPos armPos;
	bool motorsOn = false;

  //Initial arm position will be 0-0
	armPos.base = 0;
	armPos.sholder = 0;
	armPos.elbow = 0;
	armPos.wristRoll = 0;
	armPos.wristPitch = 0;
	armPos.ex1 =0;
	armPos.ex2 =0;
	scorbot->resetEncoders();
	scorbot->setArmPos(armPos);

	int currentPos = 0;
	while(1)
	{
		//sprintf(msg, "Speed = %i", speed);
		//writeText(disp, Point2D<int>(0,80), msg);

		pWin.update();

		//float gc = scorbot->gravityComp(0,0);

		// armPos = scorbot->getIKArmPos(efPos);


    




		if (showMS)
		{
			//      char b[9];
			//      scorbot->getMicroSwitchByte();
			//LINFO("MS: %s", scorbot->getMicroSwitchByte());
			//LINFO("MS: %d", scorbot->getMicroSwitch());
			printf("MS: %d | %d%d%d%d%d\n", scorbot->getMicroSwitch(), scorbot->getMicroSwitchMotor(Scorbot::BASE),
					scorbot->getMicroSwitchMotor(Scorbot::SHOLDER),
					scorbot->getMicroSwitchMotor(Scorbot::ELBOW),
					scorbot->getMicroSwitchMotor(Scorbot::WRIST_ROLL),
					scorbot->getMicroSwitchMotor(Scorbot::WRIST_PITCH));

		}


    Image<PixRGB<byte> > disp = getDisplay(motorsOn);
		ofs->writeRGB(disp, "Output", FrameInfo("output", SRC_POS));


		int key = getKey(ofs);

		Point3D<float> efPos;
		if (key != -1)
		{
			switch(key)
			{
				case 10: //1
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::BASE, speed);
					else
						scorbot->setJointPos(Scorbot::BASE, 100, true);
					break;
				case 24: //q
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::BASE, -1*speed);
					else
						scorbot->setJointPos(Scorbot::BASE, -100, true);
					break;
				case 11: //2
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::SHOLDER, speed);
					else
						scorbot->setJointPos(Scorbot::SHOLDER, 100, true);
					break;
				case 25: //w
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::SHOLDER, -1*speed);
					else
						scorbot->setJointPos(Scorbot::SHOLDER, -100, true);
					break;
				case 12: //3
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::ELBOW, -1*speed);
					else
						scorbot->setJointPos(Scorbot::ELBOW, 100, true);
					break;
				case 26: //e
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::ELBOW, speed);
					else
						scorbot->setJointPos(Scorbot::ELBOW, -100, true);
					break;
				case 13: //4
					if (!scorbot->internalPidEn())
						scorbot->setMotor(rawWrist ? Scorbot::WRIST1 : Scorbot::WRIST_PITCH, speed);
					else
						scorbot->setJointPos(rawWrist ? Scorbot::WRIST1 : Scorbot::WRIST_PITCH, 100, true);
					break;
				case 27: //r
					if (!scorbot->internalPidEn())
						scorbot->setMotor(rawWrist ? Scorbot::WRIST1 : Scorbot::WRIST_PITCH, -1*speed);
					else
						scorbot->setJointPos(rawWrist ? Scorbot::WRIST1 : Scorbot::WRIST_PITCH, -100, true);
					break;
				case 14: //5
					if (!scorbot->internalPidEn())
						scorbot->setMotor(rawWrist ? Scorbot::WRIST2 : Scorbot::WRIST_ROLL, speed);
					else
						scorbot->setJointPos(rawWrist ? Scorbot::WRIST2 : Scorbot::WRIST_ROLL, 100, true);
					break;
				case 28: //t
					if (!scorbot->internalPidEn())
						scorbot->setMotor(rawWrist ? Scorbot::WRIST2 : Scorbot::WRIST_ROLL, -1*speed);
					else
						scorbot->setJointPos(rawWrist ? Scorbot::WRIST2 : Scorbot::WRIST_ROLL, -100, true);
					break;
				case 15: //6
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::GRIPPER, speed);
					else
						scorbot->setJointPos(Scorbot::GRIPPER, 100, true);
					break;
				case 29: //y
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::GRIPPER, -1*speed);
					else
						scorbot->setJointPos(Scorbot::GRIPPER, -100, true);
					break;
				case 16: //7
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::EX1, speed);
					else
						scorbot->setJointPos(Scorbot::EX1, 300, true);
					break;
				case 30: //u
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::EX1, -1*speed);
					else
						scorbot->setJointPos(Scorbot::EX1, -300, true);
					break;
				case 17: //8
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::EX2, speed);
					else
						scorbot->setJointPos(Scorbot::EX2, 100, true);
					break;
				case 31: //i
					if (!scorbot->internalPidEn())
						scorbot->setMotor(Scorbot::EX2, -1*speed);
					else
						scorbot->setJointPos(Scorbot::EX2, -100, true);
					break;
				case 34:
					LINFO("Resetting Encoders");
					scorbot->resetEncoders();
					break;
				case 65: //space
					scorbot->stopControl();
					usleep(10000);
					scorbot->stopAllMotors();
					break;
				case 41:
				case 33: //p
					{
						Scorbot::ArmPos armPos = scorbot->readArmState();
						LINFO("Encoders: B:%d S:%d E:%d WR:%d WP:%d W1:%i W2:%i G:%d E1:%d E2:%d",
								armPos.base,
								armPos.sholder,
								armPos.elbow, 
								armPos.wristRoll,
								armPos.wristPitch,
								armPos.wrist1,
								armPos.wrist2,
								armPos.gripper,
								armPos.ex1,
								armPos.ex2);

						Point3D<float> efPos = scorbot->getEFPos(armPos);
						LINFO("EFPos(mm): %0.2f %0.2f %0.2f",efPos.x,efPos.y,efPos.z);
					}
					break;
				case 57:
					showMS = !showMS;
					break;
				case 51:
				case 58: //m
					//showMS = !showMS;
					motorsOn = !motorsOn;
					if (motorsOn)
						scorbot->motorsOn();
					else
						scorbot->motorsOff();
					break;
				case 43: //h
					scorbot->homeMotors();
					break;
				case 32://o move motor pos to 0
					//  scorbot->resetMotorPos();
					break;
				case 54://c record current motor pos
					LINFO("Saved position");
					{
						Scorbot::ArmPos armPos = scorbot->getArmPos();
						armPositions.push_back(armPos);
						//scorbot->setArmPos(armPos);
					}
					break;
				case 46://l goto the motor pos
					LINFO("Move to positions");
					for(uint i=0; i<armPositions.size(); i++)
					{
						LINFO("Going to Pos %i...", currentPos);
						scorbot->setInternalPos(armPositions[currentPos]);
					}


					break;
				case 56://b clear the record pos
					LINFO("Clear positions");
					armPositions.clear();
					break;
				case 39://s start control
					LINFO("Starting controller");
					scorbot->startControl();
					break;

				case 21: speed += 1; break; //-
				case 20: speed -= 1; break; //=

				case 79:    
								 PGain += 1000;
								 scorbot->setInternalPIDGain(pidMot, PGain, IGain, DGain);
								 LINFO("P: %d I: %d D: %d", PGain, IGain, DGain);
								 //scorbot->tunePID(pidMot, 0.001,  0, 0.0, true);
								 break;
				case 87:  
								 PGain -= 1000;
								 scorbot->setInternalPIDGain(pidMot, PGain, IGain, DGain);
								 LINFO("P: %d I: %d D: %d", PGain, IGain, DGain);
								 //scorbot->tunePID(pidMot, -0.001, 0, 0.0, true);
								 break;
				case 80: //page up
								 IGain += 1000;
								 scorbot->setInternalPIDGain(pidMot, PGain, IGain, DGain);
								 LINFO("P: %d I: %d D: %d", PGain, IGain, DGain);
								 //scorbot->tunePID(pidMot, 0, 0.001, 0, true);
								 break; 
				case 88://page down
								 IGain -= 1000;
								 scorbot->setInternalPIDGain(pidMot, PGain, IGain, DGain);
								 LINFO("P: %d I: %d D: %d", PGain, IGain, DGain);
								 //scorbot->tunePID(pidMot, 0, -0.001, 0, true);
								 break; 
				case 81:
								 DGain += 1000;
								 scorbot->setInternalPIDGain(pidMot, PGain, IGain, DGain);
								 LINFO("P: %d I: %d D: %d", PGain, IGain, DGain);
								 //scorbot->tunePID(pidMot, 0, 0.0, 0.001, true);
								 break;
				case 89:
								 DGain -= 1000;
								 scorbot->setInternalPIDGain(pidMot, PGain, IGain, DGain);
								 LINFO("P: %d I: %d D: %d", PGain, IGain, DGain);
								 //scorbot->tunePID(pidMot, 0, 0.0, -0.001, true);
								 break;

				case KEY_UP:
								 armPos = scorbot->getDesiredArmPos();
								 efPos = scorbot->getEFPos(armPos);
								 efPos.y += 5;
								 armPos = scorbot->getIKArmPos(efPos);
								 scorbot->setJointPos(Scorbot::EX1, armPos.ex1);
								 break;
				case KEY_DOWN:
								 armPos = scorbot->getDesiredArmPos();
								 efPos = scorbot->getEFPos(armPos);
								 efPos.y -= 5;
								 armPos = scorbot->getIKArmPos(efPos);
								 scorbot->setJointPos(Scorbot::EX1, armPos.ex1);
								 break;
				case KEY_LEFT:
								 armPos = scorbot->getDesiredArmPos();
								 efPos = scorbot->getEFPos(armPos);
								 efPos.x -= 5;
								 armPos = scorbot->getIKArmPos(efPos);
								 scorbot->setJointPos(Scorbot::SHOLDER, armPos.sholder);
								 scorbot->setJointPos(Scorbot::ELBOW, armPos.elbow);
								 break;
				case KEY_RIGHT:
								 armPos = scorbot->getDesiredArmPos();
								 efPos = scorbot->getEFPos(armPos);
								 efPos.x += 5;
								 armPos = scorbot->getIKArmPos(efPos);
								 scorbot->setJointPos(Scorbot::SHOLDER, armPos.sholder);
								 scorbot->setJointPos(Scorbot::ELBOW, armPos.elbow);
								 break;
				case 99: //page up
								 armPos = scorbot->getDesiredArmPos();
								 efPos = scorbot->getEFPos(armPos);
								 efPos.z += 5;
								 armPos = scorbot->getIKArmPos(efPos);
								 scorbot->setJointPos(Scorbot::SHOLDER, armPos.sholder);
								 scorbot->setJointPos(Scorbot::ELBOW, armPos.elbow);
								 break;
				case 105: //page down
								 armPos = scorbot->getDesiredArmPos();
								 efPos = scorbot->getEFPos(armPos);
								 efPos.z -= 5;
								 armPos = scorbot->getIKArmPos(efPos);
								 scorbot->setJointPos(Scorbot::SHOLDER, armPos.sholder);
								 scorbot->setJointPos(Scorbot::ELBOW, armPos.elbow);
								 break;
				case 67: //F1
							{
							}
								 break;
				case 68: //F2
								 int joint, position, duration;
								 std::cout << "Joint, Position, Duration?";
								 std::cin >> joint >> position >> duration;
								 scorbot->setJointPos((Scorbot::MOTOR)joint, position, false, duration);
								 break;
			}

			if (speed < 0) speed = 0;
			if (speed > 100) speed = 100;

			LINFO("Key = %i speed=%i", key, speed);
		}

		usleep(10000);
	}

	// stop all our ModelComponents
	manager.stop();

	// all done!
	return 0;
}
void home(nub::ref<OutputFrameSeries> &ofs)
{
	//scorbot->stopAllMotors();
	//LINFO("Start doing homeing..");
	//LINFO("Use Left/Right Key to move,S to skip,Space to stop");
	//LINFO("Move Base..");
	//calMotor(Scorbot::BASE,ofs);
	//LINFO("Move Sholder..");
	//calMotor(Scorbot::SHOLDER,ofs);
	//LINFO("Move Elbow..");
	//calMotor(Scorbot::ELBOW,ofs);
	//LINFO("Move WRIST_ROLL..");
	//calMotor(Scorbot::WRIST_ROLL,ofs);
	//LINFO("Move WRIST_PITCH..");
	//calMotor(Scorbot::WRIST_PITCH,ofs);
	//LINFO("Done, all motor @ home now...");
	//scorbot->resetEncoders();

}

//void calMotor(Scorbot::MOTOR m,nub::ref<OutputFrameSeries> &ofs, int speed)
//{
//  bool doing = true;
//  int ms = m;
//  LINFO("Move motor %d..",m);
//  while(doing)
//  {
//    int key = getKey(ofs);
//    if (key != -1)
//    {
//      switch(key)
//      {
//        case KEY_LEFT:
//          scorbot->setMotor(m, speed);
//          break;
//        case KEY_RIGHT:
//          scorbot->setMotor(m, -1*speed);
//          break;
//        case 65: //space
//          scorbot->stopAllMotors();
//          break;
//        case 39: //S,skip
//          doing = false;
//          break;
//      }
//      //LINFO("Key = %i", key);
//    }
//    if(ms==6)
//      ms = 4;
//    if(ms==7)
//      ms = 3;
//    //Wrist_ROLL = 7 which map to MS 3 & Wrist_PITCH =6 map to MS 4
//
//    //Once hit the switch,move to one side to release switch
//    if( scorbot->getMicroSwitchMotor(ms) == 0){
//      do
//      {
//        scorbot->setMotor(m, -speed);
//      }while( scorbot->getMicroSwitchMotor(ms) == 0);
//      if(ms == 4){
//        scorbot->setMotor(m, speed);
//        usleep(500000);// 1 sec
//      }
//      doing = false;
//      scorbot->stopAllMotors();
//      LINFO("MS %i is homed,plese press Space or UP/DWON",ms);
//      //if it's last motor,also open the gripper
//      if(ms==4)
//        scorbot->setMotor(Scorbot::GRIPPER, speed);
//    }
//  }
//  int key;
//  do{
//    key =getKey(ofs);
//  }while(key != 65 && key!=KEY_DOWN && key!=KEY_UP);//space
//  scorbot->stopAllMotors();
//
//
//}
// ######################################################################
/* So things look consistent in everyone's emacs... */
/* Local Variables: */
/* indent-tabs-mode: nil */
/* End: */
