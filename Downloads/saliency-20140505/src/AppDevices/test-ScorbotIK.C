#include <Ice/Ice.h>
#include "Ice/Scorbot.ice.H"
#include "Image/Point2D.H"
#include "Image/DrawOps.H"
#include "Image/MathOps.H"
#include "Image/ColorOps.H"
#include <signal.h>
#include "Component/ModelManager.H"
#include "Raster/Raster.H"
#include "Util/MathFunctions.H"
#include "Util/Types.H"
#include "Util/log.H"
#include "Media/FrameSeries.H"
#include "Transport/FrameInfo.H"
#include "Raster/GenericFrame.H"
#include "GUI/XWinManaged.H"
#include "GUI/ImageDisplayStream.H"
#include "GUI/PrefsWindow.H"
#include "GUI/DebugWin.H"


Robots::ScorbotIcePrx scorbot;

// ######################################################################
void cleanupAndExit()
{
  std::cerr <<
    std::endl << "*** EXITING - STOPPING SCORBOT ***" << std::endl;
  scorbot->stopAllMotors();
  sleep(1);
  exit(0);
}

// ######################################################################
void terminate(int s)
{
  cleanupAndExit();
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



Point2D<double> getTargetPos()
{
  double x, y;

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Enter a target point in mm: (x y): ";
  std::cin >> x >> y;
  return Point2D<double>(x,y);
}

// ######################################################################
// Point the camera at a 2D position on the board at a given angle (camera to board angle, in radians), with the camera
// 'camDist' mm away from the target, and the linear slide set xOffset mm to the side of the target.
Robots::ArmPos calcArmPos(Point2D<double> targetPos, double xOffset, double angle, double camDist)
{
  //////////////////////////////////////////////////////////////////////// 
  // CONSTANTS
  double armSlideXoffset = 0.0; // The x-offset from origin to the arm's home slide position
  double armSlideYoffset = 0.0; // The y-offset from the origin to the arm's slide traversal axis (this should be a neg. number)
  double armZoffset      = 0.0; // The distance from the arm's origin to the table surface

  //These offsets assume that when the wrist is pointed straight out, the x-axis points through the tip of the gripper,
  //and the y-axis points towards the ceiling. 
  double camToWristX     = 0.0; // The distance from the cam center to the wrist along the x-axis if the wrist is straight out
  double camToWristY     = 0.0; // The distance from the cam center to the wrist along the y-axis if the wrist is straight out
  //
  //////////////////////////////////////////////////////////////////////// 

  double targetX = targetPos.i;
  double targetY = targetPos.j;

  // Distance from the arm center to the target in the x-y (tabletop) plane
  double d_t = sqrt(pow(xOffset,2) + pow(targetY+armSlideYoffset,2));

  // Relative position of the camera center from the target
  double x_hat_cam = cos(angle)*camDist;
  double y_hat_cam = sin(angle)*camDist;

  // Position of the camera relative to the robot origin.
  // This position is in an artificial coord. system created by assuming the
  // arm is already at it's proper slide offset, and the base has been rotated
  // to the target
  double x_cam = d_t - x_hat_cam;
  double y_cam = y_hat_cam - armZoffset;

  // Rotate the camera-to-wrist offsets according to the desired angle
  double cam_x_off = camToWristX * cos(-angle) - camToWristY*sin(-angle);
  double cam_y_off = camToWristX * sin(-angle) + camToWristY*cos(-angle);

  // Absolute wrist position (in our artificial coord. system)
  double wrist_x = x_cam + cam_x_off;
  double wrist_y = y_cam + cam_y_off;

  // Absolute global slider position
  double sliderPos =  targetX + xOffset - armSlideXoffset;

  // Base angle
  double baseAngle = atan2(xOffset, targetY+armSlideYoffset);

  // Get the inverse kinematics from the scorbot
  Robots::ArmPos ik = scorbot->getIK(0, wrist_x, wrist_y); 

  // Force the slide position and base angle
  ik.ex1  = scorbot->mm2enc(sliderPos); 
  ik.base = scorbot->ang2enc(baseAngle);

  std::cout << "---IK---" << std::endl;
  std::cout << "base:     " << ik.base << std::endl;
  std::cout << "shoulder: " << ik.shoulder << std::endl;
  std::cout << "elbow:    " << ik.elbow << std::endl;
  std::cout << "wrist1:   " << ik.wrist1 << std::endl;
  std::cout << "wrist2:   " << ik.wrist2 << std::endl;
  std::cout << "ex1:      " << ik.ex1 << std::endl;
  std::cout << "--------" << std::endl;

  return ik;
}



std::vector<Robots::ArmPos> getArmPositions()
{

	std::vector<Robots::ArmPos> positions;
	Robots::ArmPos armpos;
	armpos.base = 0;
	armpos.shoulder = 0;
	armpos.elbow = 0;
	armpos.wrist1 = 0;
	armpos.wrist2 = 0;
	armpos.ex1 = 0;
	armpos.ex2 = 0;
	armpos.duration = 500;

	armpos.base=-823; armpos.shoulder=1594; armpos.elbow=-26751; armpos.wrist1=224; armpos.wrist2=-125; armpos.gripper=0; armpos.ex1=49; 
	positions.push_back(armpos); 
	armpos.base=-707; armpos.shoulder=1292; armpos.elbow=-27673; armpos.wrist1=1143; armpos.wrist2=-918; armpos.gripper=0; armpos.ex1=49;
	positions.push_back(armpos); 



/*
  //armpos.base=-4253; armpos.shoulder=-1204; armpos.elbow=413; armpos.wrist1=618; armpos.wrist2=-937; armpos.gripper=0; armpos.ex1=4915; 
  armpos.ex1=4915; 
	positions.push_back(armpos); 
  //armpos.base=590; armpos.shoulder=-1205; armpos.elbow=857; armpos.wrist1=615; armpos.wrist2=-795; armpos.gripper=0; armpos.ex1=95684; 
  armpos.ex1=95684; 
	positions.push_back(armpos); 
  //armpos.base=3140; armpos.shoulder=-1204; armpos.elbow=856; armpos.wrist1=615; armpos.wrist2=-795; armpos.gripper=0; armpos.ex1=141287;
  armpos.ex1=141287;
	positions.push_back(armpos); 
  //armpos.base=3322; armpos.shoulder=-9856; armpos.elbow=1556; armpos.wrist1=348; armpos.wrist2=-332; armpos.gripper=0; armpos.ex1=141271;
  armpos.ex1=141271;
	positions.push_back(armpos); 
  //armpos.base=-1418; armpos.shoulder=-8198; armpos.elbow=6136; armpos.wrist1=1152; armpos.wrist2=-1106; armpos.gripper=0; armpos.ex1=58362;
  armpos.ex1=58362;
	positions.push_back(armpos); 
  //armpos.base=-4147; armpos.shoulder=-11764; armpos.elbow=101; armpos.wrist1=489; armpos.wrist2=-584; armpos.gripper=0; armpos.ex1=4931; 
  armpos.ex1=4931; 
	positions.push_back(armpos); 
*/

  return positions;

}

void setWrist(Robots::ArmPos &armPos, float ang)
{
   armPos.wrist1 = (ang*(2568.0*2.0)/M_PI);
   armPos.wrist2 = -1*ang*(2568.0*2.0)/M_PI;
}

// ######################################################################
int main(int argc, char* argv[])
{
	signal(SIGHUP, terminate); signal(SIGINT, terminate);
	signal(SIGQUIT, terminate); signal(SIGTERM, terminate);
	signal(SIGALRM, terminate);


	ModelManager manager("Test Model for Scorbot robot arm controller");

	nub::ref<OutputFrameSeries> ofs(new OutputFrameSeries(manager));
	manager.addSubComponent(ofs);

  nub::ref<InputFrameSeries> ifs(new InputFrameSeries(manager));
  manager.addSubComponent(ifs);


	// Parse command-line:
	if (manager.parseCommandLine(argc, argv, "FileName", 0, 1) == false) return(1);

	std::string fileName = manager.getExtraArg(0);

	// let's get all our ModelComponent instances started:
	manager.start();

  ifs->startStream();
  while(1)
  {
	  GenericFrame input = ifs->readFrame();
	  if (!input.initialized())
		  break;


	  Image<PixRGB<byte> > inImage = input.asRgb();
	  Point2D<int> center(inImage.getWidth()/2, inImage.getHeight()/2);
	  Dims winSize(512,512);

	  Image<PixRGB<byte> > objImg = crop(inImage, Rectangle::centerDims(center, winSize));
	  drawCircle(objImg, Point2D<int>(256,256), 3, PixRGB<byte>(255,0,0), 3);

	  drawCircle(inImage, Point2D<int>(inImage.getWidth()/2,inImage.getHeight()/2), 3, PixRGB<byte>(255,0,0), 3);

	  ofs->writeRGB(inImage, "Output", FrameInfo("output", SRC_POS));

  }

/*
  try {
    ic = Ice::initialize(argc, argv);
    Ice::ObjectPrx base = ic->stringToProxy(
        "ScorbotServer:default -p 10000 -h ihead");
    scorbot = Robots::ScorbotIcePrx::checkedCast(base);
    if(!scorbot)
      throw "Invalid Proxy!";


		std::vector<Robots::ArmPos> armPos =  getArmPositions();
		Robots::ArmPos homePos = armPos[0];
    //homePos.base = 0; homePos.shoulder = 0; homePos.elbow = 0; homePos.wrist1 = 0; homePos.wrist2 = 0; homePos.ex1 = 0; homePos.ex2 = 0;
    //homePos.duration = 1000;

	
 
		LINFO("Move robot %i %i", homePos.wrist1, homePos.wrist2);
		//scorbot->setArmPos(homePos);
		//getchar();
    	
		scorbot->setArmPos(armPos[0]);
		sleep(2);
    LINFO("Done");

    uint currentArmPos = 0;
	  bool motorsOn = false;
		bool clearToShoot = false;
		bool startSequance = false;
     
		Image<PixRGB<byte> > lastInput;

    while(1)
		{
			const FrameState is = ifs->updateNext();
			if (is == FRAME_COMPLETE)
				break;

			GenericFrame input = ifs->readFrame();
			if (!input.initialized())
				break;


			Image<PixRGB<byte> > inImage = input.asRgb();
			Point2D<int> center(inImage.getWidth()/2, inImage.getHeight()/2);
			Dims winSize(512,512);

			Image<PixRGB<byte> > objImg = crop(inImage, Rectangle::centerDims(center, winSize));


			double imgSum = 1e10;
			if (lastInput.initialized())
			{
				Image<float> lum1 = luminance(objImg);
				Image<float> lum2 = luminance(lastInput);
				Image<float> diffImg = lum1 - lum2;
				imgSum = sum(diffImg);
			}
			if (!(ifs->frame()%10))
			{
				lastInput = objImg;
				double diff = fabs(imgSum)/(512*512); 

				int time = scorbot->getMovementTime(); 
				//LINFO("Going to pos %i at %i/%i", currentArmPos, time, armPos[currentArmPos].duration);

				if (diff < 0.5 && time >= armPos[currentArmPos].duration)
					clearToShoot = true;
			}		

			

			drawRect(inImage, Rectangle::centerDims(center, winSize), PixRGB<byte>(0,255,0));
			drawCircle(inImage, center, 3, PixRGB<byte>(254,0,0), 3);




			if (startSequance)
			{
				if (clearToShoot)
				{
					char file[255];
					sprintf(file, "%s/pos%i.ppm", fileName.c_str(), currentArmPos); //This is the previus position
					//LINFO("Capture image to %s", file);
					sleep(2);
					//Raster::WriteRGB(objImg, file);
					//SHOWIMG(objImg);
				
					clearToShoot = false;

					Robots::ArmPos curArmPos = scorbot->getArmPos();
					Robots::ArmPos desArmPos = armPos[currentArmPos];
					LINFO ("armpos.base=%i/%i; armpos.shoulder=%i/%i; armpos.elbow=%i/%i; armpos.wrist1=%i/%i; armpos.wrist2=%i/%i;",
						curArmPos.base, desArmPos.base,
						curArmPos.shoulder, desArmPos.shoulder,
						curArmPos.elbow, desArmPos.elbow,
						curArmPos.wrist1, desArmPos.wrist1,
						curArmPos.wrist2, desArmPos.wrist2);

					scorbot->setArmPos(armPos[currentArmPos]);
					currentArmPos++;
					if (currentArmPos > armPos.size())
					{
						currentArmPos = 0;
						startSequance = true;
						scorbot->setArmPos(armPos[currentArmPos]);
					}
				}

			}


			drawCircle(objImg, Point2D<int>(256,256), 3, PixRGB<byte>(255,0,0), 3);
			ofs->writeRGB(objImg, "Output", FrameInfo("output", SRC_POS));


			int key = getKey(ofs);

			if (key != -1)
			{
				LINFO("Key %i", key);
				switch(key)
				{
						case 36: //enter
							startSequance = true;
							break;
						case 43: //h home
							LINFO("Going home\n");
							scorbot->setArmPos(homePos);
							break;
						case 33: //p getPositions
							{
								Robots::ArmPos armpos = scorbot->getArmPos();
								LINFO ("armpos.base=%i; armpos.shoulder=%i; armpos.elbow=%i; armpos.wrist1=%i; armpos.wrist2=%i; armpos.gripper=%i; armpos.ex1=%i; armpos.ex2=%i;",
								armpos.base, armpos.shoulder, armpos.elbow , armpos.wrist1, armpos.wrist2, armpos.gripper, armpos.ex1, armpos.ex2);
							}
							break;
						case 58: //m
							motorsOn = !motorsOn;
							if (motorsOn)
								scorbot->motorsOn();
							else
								scorbot->motorsOff();
							break;
				}
			}

		}
    //mainLoop();

  } catch(const Ice::Exception& ex) {
    std::cerr << ex << std::endl;
    status = 1;
  } catch(const char* msg) {
    std::cerr << msg << std::endl;
    status = 1;
  }

  if(ic)
    ic->destroy();

  cleanupAndExit();
  return status;
*/
return 0;
}
